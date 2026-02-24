# auto

Contains the inference algorithm implementations that orchestrate the full control loop: reading robot state, running policy inference, and publishing actions.

## Structure

```
auto/
└── inference_algorithms/
    ├── rtc/          # Real-Time Control (dual-process, shared memory)
    └── sequential/   # Sequential inference (single-threaded)
```

See `inference_algorithms/README.md` for detailed documentation of each algorithm.
