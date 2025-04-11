# %%
"""
https://chatgpt.com/c/67cb8388-7750-8004-be75-8c6249735430
probably a better version

https://threadreaderapp.com/thread/1816459931995840984.html
argues that
"we observate high level math to be pareto distributed (eg. Terrance Tao v everyone"), if math was mostly genetic it'd be normal (there's few major genes, just small linear additive effects), therefore high level math isn't mostly genetic"

The problem is looking at the top 10% or top 1% it's hard to distinguish a pareto from a log-normal distribution, and the assumption its pareto is suspect.

Eg if we assume high level math performance is the product of 3-7 features each correlated at 0.2-0.7, and we only care about top 10% then it's basically pareto.
We can't tell the difference with the number of datapoints available to us.

Since there's about 80k professional mathematicians worldwide
About 500k math PhD's in the world
At most 4 million Math BAs in the world.
    Best PhD acceptance rates around 10%, though most higher, so lines up.
So when looking at high level math performance, we have 5 million math BAs, and we base our observations on the ability of math PhDs and above.

We want to know if we can tell if it's log-normal or pareto from the top 10% of samples from a distribution of 4 million.

The data is generated from a log normal distribution, can we tell it's not Pareto?
    At p=0.05 we can't reject the data is pareto distributed when [(cov=0.2, dim=15), (cov=0.5,dim=9), (cov=0.7, dim=9)]
    But looking at the graphs, it all seems functionally pareto. I don't think recolloecting from personal experiance would be able to tell you.
    And If we started with only the Professional mathematicision n=80k, and looked at the very best it's ['cov=0.2, dim=9', 'cov=0.2, dim=15', 'cov=0.5, dim=9', 'cov=0.5, dim=15', 'cov=0.7, dim=5', 'cov=0.7, dim=9', 'cov=0.7, dim=15']
Not really.

So while the argument works for ruling out the effect of the predictiveness of general intelligence, it doesn't rule out genes.
It started by falsely assuming math talent is pareto distributed when it could actually be log-normal.

But use https://arxiv.org/abs/0706.1062 for fitting to power law data instead?
https://gwern.net/doc/iq/ses/1957-shockley.pdf example of multiplication of abilities. But difference in "research value" and "raw math problem solving value"

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, pareto, lognorm, cramervonmises

np.random.seed(0)
# N = 4_000_000  # math BAs
N = 80_000  # professional mathematicians
p_cutoff = 0.05  # p-value above which can't prove they're different so generate graphs
top_percent = 50  # what % of ability to graph
cant_reject = []
for cov in [0.2, 0.5, 0.7]:
    dims = np.unique(np.round(np.logspace(np.log10(2), np.log10(15), num=5)).astype(int))
    std = 0.1

    for d in dims:
        # Construct the mean vector and covariance matrix with 50% off-diagonals.
        mean_vector = np.ones(d)
        cov_matrix = np.full((d, d), cov * std**2)
        np.fill_diagonal(cov_matrix, std**2)

        # Sample correlated traits from a multivariate normal.
        traits = np.random.multivariate_normal(mean_vector, cov_matrix, size=N)
        ability = np.prod(traits, axis=1)

        # Focus on the top 10% of ability scores.
        threshold = np.percentile(ability, 100 - top_percent)
        top10 = ability[ability >= threshold]
        print(
            np.percentile(ability, 99),
            np.percentile(ability, 99.9),
            np.percentile(ability, 99.99),
            np.percentile(ability, 99.999),
            max(ability),
        )

        # Fit distributions to the top10 data.
        norm_mu, norm_sigma = norm.fit(top10)
        pareto_shape, pareto_loc, pareto_scale = pareto.fit(top10)
        lognorm_shape, lognorm_loc, lognorm_scale = lognorm.fit(top10, floc=0)

        # Compute Cramér–von Mises tests for each fitted distribution.
        cvm_norm = cramervonmises(top10, lambda x: norm.cdf(x, norm_mu, norm_sigma))
        cvm_pareto = cramervonmises(
            top10, lambda x: pareto.cdf(x, pareto_shape, pareto_loc, pareto_scale)
        )
        cvm_lognorm = cramervonmises(
            top10, lambda x: lognorm.cdf(x, lognorm_shape, lognorm_loc, lognorm_scale)
        )

        print(f"\nDimension = {d}")
        print(
            f"CVM Test - Normal fit: statistic={cvm_norm.statistic:.4f},"
            f" p-value={cvm_norm.pvalue:.4f}"
        )
        print(
            f"CVM Test - Pareto fit: statistic={cvm_pareto.statistic:.4f},"
            f" p-value={cvm_pareto.pvalue:.4f}"
        )
        print(
            f"CVM Test - Lognormal fit: statistic={cvm_lognorm.statistic:.4f},"
            f" p-value={cvm_lognorm.pvalue:.4f}"
        )
        if cvm_pareto.pvalue >= p_cutoff:
            cant_reject += [(cov, d)]

        # Generate x-axis values for PDF plotting.
        x = np.linspace(top10.min(), top10.max(), 1000)
        pdf_norm = norm.pdf(x, loc=norm_mu, scale=norm_sigma)
        pdf_pareto = pareto.pdf(x, pareto_shape, loc=pareto_loc, scale=pareto_scale)
        pdf_lognorm = lognorm.pdf(x, lognorm_shape, loc=lognorm_loc, scale=lognorm_scale)

        # Plot histogram and fitted PDFs with white background and dark text.
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
        ax.set_facecolor("white")

        # Use a light gray color for the histogram bars so that text and lines stand out.
        ax.hist(
            top10,
            bins=50,
            density=True,
            alpha=0.5,
            label=f"Top {top_percent}% Data",
            color="lightgray",
            edgecolor="black",
        )
        ax.plot(x, pdf_norm, "r-", label=f"Normal fit\nμ={norm_mu:.2f}, σ={norm_sigma:.2f}")
        ax.plot(x, pdf_pareto, "g-", label=f"Pareto fit\nshape={pareto_shape:.2f}")
        ax.plot(
            x,
            pdf_lognorm,
            "b-",
            label=f"Lognormal fit\nshape={lognorm_shape:.2f}, scale={lognorm_scale:.2f}",
        )

        # Set labels and title with dark text.
        ax.set_xlabel("Ability", color="black")
        ax.set_ylabel("Density", color="black")
        ax.set_title(
            f"Fitted Distributions for Top {top_percent}% Ability\n(Product of {d},"
            f" {cov} Correlated Normals)",
            color="black",
        )

        # Use a legend with a white background and dark border.
        leg = ax.legend(facecolor="white", edgecolor="black")
        plt.tight_layout()
        plt.show()

        print([f"cov={i[0]}, dim={i[1]}" for i in cant_reject])
