#!/bin/bash
# start_ray.sh — run on each machine

# HEAD_IP="192.168.0.134"
# HOSTNAME=$(hostname)

# case "$HOSTNAME" in
#   robros-ai1)
#     ray start --head --port=6379
#     ;;
#   robros-5090)
#     ray start --address=${HEAD_IP}:6379 \
#       --resources='{"inference_pc": 1}'
#     ;;
#   *)
#     echo "Unknown host: $HOSTNAME"
#     exit 1
#     ;;
# esac

#!/bin/bash
# start_ray.sh — run on each machine

# Use the 100.x network IP of robros-ai1 (Head node)
HEAD_IP="100.109.184.39"
HOSTNAME=$(hostname)

case "$HOSTNAME" in
  robros-ai1)
    echo "Starting Ray Head Node on $HOSTNAME..."
    # Explicitly bind to ai1's VPN IP so it doesn't pick the Docker interface
    ray start --head \
      --port=6379 \
      --node-ip-address=${HEAD_IP} \
      --dashboard-host=0.0.0.0
    ;;
  robros-5090)
    echo "Starting Ray Worker Node on $HOSTNAME..."
    # Explicitly bind to the 5090's VPN IP
    WORKER_IP="100.112.232.50"
    ray start --address=${HEAD_IP}:6379 \
      --node-ip-address=${WORKER_IP} \
      --resources='{"inference_pc": 1}'
    ;;
  *)
    echo "Unknown host: $HOSTNAME"
    exit 1
    ;;
esac