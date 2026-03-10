import torch

CKPT = "crouch.pt"  # change


def main():
    ckpt = torch.load(CKPT, map_location="cpu")
    sd = ckpt["model_state_dict"]

    print("Loaded keys:", ckpt.keys())
    print("State dict params:", len(sd))
    print("\nSample state_dict keys:")
    for k in list(sd.keys()):
        print(" ", k)

    # Import ActorCritic (path varies by install)
    try:
        from rsl_rl.modules import ActorCritic
    except Exception:
        from rsl_rl.modules.actor_critic import ActorCritic

    num_obs = 45
    num_actions = 12

    # MUST match your training config
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"
    init_noise_std = 1.0

    # ---- YOUR VERSION NEEDS num_actor_obs and num_critic_obs ----
    ac = ActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        activation=activation,
        init_noise_std=init_noise_std,
    )

    missing, unexpected = ac.load_state_dict(sd, strict=False)
    print("\nLoaded into ActorCritic.")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    print("\n=== ActorCritic module ===")
    print(ac)

    # ---- Check deterministic inference output range ----
    ac.eval()
    obs = torch.randn(4096, num_obs)

    with torch.no_grad():
        if hasattr(ac, "act_inference"):
            a = ac.act_inference(obs)
            used = "act_inference"
        else:
            # Fallback: actor forward (NN output (usually mean action)
            a = ac.actor(obs)
            used = "actor(obs)"

    print(f"\nUsed: {used}")
    print("Action shape:", tuple(a.shape))
    print("Action min/max:", float(a.min()), float(a.max()))
    print("Action mean/std:", float(a.mean()), float(a.std()))

    # ---- Check if tanh module exists ----
    has_tanh = any(m.__class__.__name__.lower() == "tanh" for m in ac.modules())
    print("\nHas nn.Tanh module in ActorCritic?", has_tanh)


if __name__ == "__main__":
    main()
