from engine.policies.registry import POLICY_REGISTRY
from engine.policies.interfaces import PolicyBuilder, InferencePolicy
from engine.policies.normalization import load_stats
import torch.nn as nn
import torch
import einops
from torchdiffeq import odeint as odeint_solver

class NaiveFlowMatchingPolicy(nn.Module):
    def __init__(self, components, params, normalization):
        super().__init__()
        self.num_ode_sim_steps = params.num_ode_sim_steps

        self.action_decoder = components['action_decoder']
        self.vision_encoder = components['backbone']
        self.depth_encoder = components['da3']
        self.proprio_projector = components['proprio_projector']
        

        self.state_dim = params.state_dim
        self.action_dim = params.action_dim
        self.num_queries = params.num_queries
        self.num_robot_observations = params.num_robot_observations
        self.num_image_observations = params.num_image_observations
        self.image_observation_skip = params.image_observation_skip
        self.camera_names = list(params.camera_names)
        self._sm, self._ss, self._am, self._asd, self.stats_eps = normalization
    
    @torch.inference_mode()
    def forward(self, rh: torch.Tensor, cam_images: torch.Tensor, noise: torch.Tensor, ):
        data = self.org_data(rh, cam_images)
        """ Backbone """
        with torch.no_grad():
            """ Radiov3 """
            head_image_features, head_image_semantic = self.vision_encoder(data['observation.images.cam_head'])
            head_image_features = einops.rearrange(head_image_features, 'b c h w -> b 1 c h w')

            left_image_features, _ = self.vision_encoder(data['observation.images.cam_left'])
            left_image_features = einops.rearrange(left_image_features, 'b c h w -> b 1 c h w')

            right_image_features, _ = self.vision_encoder(data['observation.images.cam_right'])
            right_image_features = einops.rearrange(right_image_features, 'b c h w -> b 1 c h w')

            """ Depth """
            # outputs (batch, num_features, height, width, feature_dim) shaped latent features
            depth_head = self.depth_encoder(image=data['observation.images.cam_head'], export_feat_layers=[8, 13, 18, 23])
            depth_left = self.depth_encoder(image=data['observation.images.cam_left'], export_feat_layers=[8, 13, 18, 23])
            depth_right = self.depth_encoder(image=data['observation.images.cam_right'], export_feat_layers=[8, 13, 18, 23])

        """ Proprio Projection """
        # Assumes that proprio feature dimension will be matched to that of visual

        conditioning_info = self.proprio_projector(cond_proprio=data['observation.state'],
                                                    cond_visual=torch.cat([head_image_features,
                                                                            left_image_features, 
                                                                            right_image_features],
                                                                            dim=1),)
        conditioning_memory = torch.cat([einops.rearrange(depth_head, 'b n h w d -> b (n h w) d'),
                                        einops.rearrange(depth_left, 'b n h w d -> b (n h w) d'),
                                        einops.rearrange(depth_right, 'b n h w d -> b (n h w) d'),
                                        conditioning_info], dim=1)
        
        batch_size = conditioning_memory.shape[0]
        device = conditioning_memory.device
        dtype = conditioning_memory.dtype

        """ Flow Matching """
        with torch.no_grad():
            actions_hat = noise
            time_grid = torch.linspace(
                0.0,
                1.0,
                self.num_ode_sim_steps + 1,
                device=conditioning_memory.device,
                dtype=conditioning_memory.dtype,
            )

            def ode_rhs(t, x):
                if t.dim() == 0:
                    t_batch = torch.full((batch_size,), t, device=device, dtype=dtype)
                else:
                    t_batch = t.to(device=device, dtype=dtype)
                    if t_batch.shape[0] != batch_size:
                        t_batch = t_batch.expand(batch_size)
                
                return self.action_decoder(
                        time=t_batch,
                        noise=x,
                        memory_input=conditioning_memory,
                        discrete_semantic_input=head_image_semantic,)

            step_size = 1.0 / float(self.num_ode_sim_steps)
            sol = odeint_solver(
                ode_rhs,
                actions_hat,
                time_grid,
                method="midpoint",
                options={"step_size": step_size},
            )
            actions_hat = sol[-1]

        return actions_hat

    @property
    def normalization_tensors(self):
        return self._sm, self._ss, self._am, self._asd, self.stats_eps
    

    def org_data(self, rh, cam_images):
        return {
            'observation.state': rh,
            'observation.images.cam_head': cam_images[:, 0, 0, :, :, :],
            'observation.images.cam_left': cam_images[:, 1, 0, :, :, :],
            'observation.images.cam_right': cam_images[:, 2, 0, :, :, :],
        }

    # For real_time_action_chunking you must expose encode_vision and body
    @torch.inference_mode()
    def encode_memory(self, rh: torch.Tensor, cam_images: torch.Tensor):
        data = self.org_data(rh, cam_images)
        """ Backbone """
        with torch.no_grad():
            """ Radiov3 """
            head_image_features, head_image_semantic = self.vision_encoder(data['observation.images.cam_head'])
            head_image_features = einops.rearrange(head_image_features, 'b c h w -> b 1 c h w')

            left_image_features, _ = self.vision_encoder(data['observation.images.cam_left'])
            left_image_features = einops.rearrange(left_image_features, 'b c h w -> b 1 c h w')

            right_image_features, _ = self.vision_encoder(data['observation.images.cam_right'])
            right_image_features = einops.rearrange(right_image_features, 'b c h w -> b 1 c h w')

            """ Depth """
            # outputs (batch, num_features, height, width, feature_dim) shaped latent features
            depth_head = self.depth_encoder(image=data['observation.images.cam_head'], export_feat_layers=[8, 13, 18, 23])
            depth_left = self.depth_encoder(image=data['observation.images.cam_left'], export_feat_layers=[8, 13, 18, 23])
            depth_right = self.depth_encoder(image=data['observation.images.cam_right'], export_feat_layers=[8, 13, 18, 23])

        """ Proprio Projection """
        # Assumes that proprio feature dimension will be matched to that of visual
        conditioning_info = self.proprio_projector(cond_proprio=data['observation.state'],
                                                    cond_visual=torch.cat([head_image_features,
                                                                            left_image_features, 
                                                                            right_image_features],
                                                                            dim=1),)
        
        return  torch.cat([einops.rearrange(depth_head, 'b n h w d -> b (n h w) d'),
                        einops.rearrange(depth_left, 'b n h w d -> b (n h w) d'),
                        einops.rearrange(depth_right, 'b n h w d -> b (n h w) d'),
                        conditioning_info], dim=1),\
                head_image_semantic

    @property
    def body(self):
        return self.action_decoder






class NaiveFlowMatchingPolicyBuilder(PolicyBuilder):
    def build_policy(self, components, params, device=None):
        normalization = load_stats(params.stats_path, params.state_dim, params.action_dim, device, params.stats_eps)
        policy = NaiveFlowMatchingPolicy(components, params, normalization)
        if device is not None:
            policy.to(device)
        return policy

POLICY_REGISTRY.add("naive_flow_matching_policy", NaiveFlowMatchingPolicyBuilder())