from __future__ import annotations

import numpy as np

from ai_books_tracking.scripts.temp import check_latent_rerating_uncertainty as module


def test_estimate_noise_model_from_pairs_recovers_components() -> None:
    rng = np.random.default_rng(0)
    true_mu = 3.1
    true_between_sd = 0.9
    true_within_sd = 0.4
    pass2_bias = 0.2

    latent = rng.normal(true_mu, true_between_sd, 20_000)
    pass1 = latent + rng.normal(0.0, true_within_sd, len(latent))
    pass2 = latent + pass2_bias + rng.normal(0.0, true_within_sd, len(latent))

    model = module.estimate_noise_model_from_pairs("enjoy", pass1, pass2)

    assert abs(model.mu - true_mu) < 0.03
    assert abs(model.pass2_bias - pass2_bias) < 0.03
    assert abs(model.within_sd - true_within_sd) < 0.03
    assert abs(model.between_sd - true_between_sd) < 0.03


def test_posterior_latent_params_shrink_and_reduce_variance_with_more_observations() -> (
    None
):
    noise_model = module.NoiseModel(
        target="enjoy",
        n_double_rated=100,
        mu=3.0,
        pass2_bias=0.0,
        within_sd=0.6,
        between_sd=1.0,
        reliability_single=1.0 / (1.0 + 0.36),
        reliability_avg2=1.0 / (1.0 + 0.18),
    )

    mean_1, sd_1 = module.posterior_latent_params(4.0, 1, noise_model)
    mean_2, sd_2 = module.posterior_latent_params(4.0, 2, noise_model)

    assert 3.0 < mean_1 < 4.0
    assert mean_2 > mean_1
    assert sd_2 < sd_1
