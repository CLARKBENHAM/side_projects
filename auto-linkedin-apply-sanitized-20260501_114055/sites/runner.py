"""Main runner - orchestrates applying across all job sites.

Usage:
    python -m sites.runner                    # run all sites
    python -m sites.runner indeed dice        # run specific sites
    python -m sites.runner --list             # list available sites
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from sites.base import ApplyResults, SiteApplier
from sites.config import ApplicantProfile, SearchConfig
from sites.indeed import IndeedApplier
from sites.dice import DiceApplier
from sites.ziprecruiter import ZipRecruiterApplier
from sites.wellfound import WellfoundApplier

SITE_REGISTRY: dict[str, type[SiteApplier]] = {
    "indeed": IndeedApplier,
    "dice": DiceApplier,
    "ziprecruiter": ZipRecruiterApplier,
    "wellfound": WellfoundApplier,
}

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"logs/sites_{timestamp}.log"),
        ],
    )
    # Quiet noisy libraries
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def run_sites(site_names: list[str] | None = None) -> dict[str, ApplyResults]:
    """Run appliers for the specified sites (or all if None)."""
    profile = ApplicantProfile()
    search_config = SearchConfig()
    results: dict[str, ApplyResults] = {}

    sites_to_run = site_names or list(SITE_REGISTRY.keys())

    for site_name in sites_to_run:
        if site_name not in SITE_REGISTRY:
            logger.error(
                "Unknown site: %s (available: %s)",
                site_name,
                list(SITE_REGISTRY.keys()),
            )
            continue

        applier_cls = SITE_REGISTRY[site_name]
        applier = applier_cls(profile=profile, search_config=search_config)

        logger.info("=" * 60)
        logger.info("Starting %s", site_name.upper())
        logger.info("=" * 60)

        try:
            result = applier.run()
            results[site_name] = result
        except KeyboardInterrupt:
            logger.info("Interrupted by user, moving to next site...")
            results[site_name] = applier.results
        except Exception:
            logger.exception("Fatal error on %s", site_name)
            results[site_name] = applier.results
        finally:
            applier.quit()

    # Print combined summary
    _print_combined_summary(results)
    _save_session_summary(results)
    return results


def _print_combined_summary(results: dict[str, ApplyResults]) -> None:
    total_applied = sum(r.applied for r in results.values())
    total_failed = sum(r.failed for r in results.values())
    total_skipped = sum(r.skipped for r in results.values())
    total_networked = sum(r.networked for r in results.values())

    logger.info("\n" + "=" * 60)
    logger.info("COMBINED RESULTS ACROSS ALL SITES")
    logger.info("=" * 60)
    for name, r in results.items():
        logger.info(
            "  %-15s applied=%-3d failed=%-3d skipped=%-3d network=%-3d",
            name,
            r.applied,
            r.failed,
            r.skipped,
            r.networked,
        )
    logger.info("-" * 60)
    logger.info(
        "  %-15s applied=%-3d failed=%-3d skipped=%-3d network=%-3d",
        "TOTAL",
        total_applied,
        total_failed,
        total_skipped,
        total_networked,
    )


def _save_session_summary(results: dict[str, ApplyResults]) -> None:
    date_str = datetime.now().strftime("%Y_%m_%d")
    summary_path = f"session_sites_{date_str}.md"

    lines = [
        f"# Multi-Site Job Applier Session - {datetime.now().strftime('%B %d, %Y')}\n",
        "## Results by Site\n",
        "| Site | Attempted | Applied | Failed | Skipped | Networking |",
        "|------|-----------|---------|--------|---------|------------|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r.attempted} | {r.applied} | {r.failed} | {r.skipped} | {r.networked} |"
        )

    total_applied = sum(r.applied for r in results.values())
    total_networked = sum(r.networked for r in results.values())
    lines.append(
        f"\n**Total applied: {total_applied}, saved for networking: {total_networked}**\n"
    )
    lines.append("## Files\n")
    lines.append("- Networking jobs: `all excels/networking_jobs.csv`")
    for name in results:
        lines.append(f"- {name} history: `all excels/{name}_applied_history.csv`")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Session summary saved to %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply to jobs across multiple sites")
    parser.add_argument("sites", nargs="*", help="Sites to run (default: all)")
    parser.add_argument("--list", action="store_true", help="List available sites")
    args = parser.parse_args()

    if args.list:
        print("Available sites:", ", ".join(SITE_REGISTRY.keys()))
        return

    setup_logging()
    run_sites(args.sites or None)


if __name__ == "__main__":
    main()
