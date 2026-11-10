"""Structural-search choice shared by CFG and standalone synthesis."""

SEARCH_STRATEGIES = ("mcmc", "enumeration", "contract")


def add_search_options(parser):
    parser.add_argument(
        "--search-strategy",
        choices=SEARCH_STRATEGIES,
        default="mcmc",
        help="Structural candidates: MCMC mutations, enumeration, or contract-guided beam search",
    )


def validate_search_strategy(strategy):
    if strategy not in SEARCH_STRATEGIES:
        raise ValueError("Search strategy must be mcmc, enumeration or contract")
    return strategy


def add_contract_search_options(parser):
    """Proposal budgets are separate from rollout/CEM and distance scales."""
    group = parser.add_argument_group("contract-guided proposals")
    group.add_argument("--contract-beam-width", type=int, default=24)
    group.add_argument("--contract-max-expansions", type=int, default=300)
    group.add_argument("--contract-seconds", type=float, default=35.0)
    group.add_argument("--contract-pool-limit", type=int, default=12)
    group.add_argument(
        "--contract-grid-only",
        action="store_true",
        help="Omit observation-derived offsets; retain demonstration ranking",
    )
    group.add_argument("--contract-no-memo", action="store_true")
    group.add_argument("--contract-no-diversity", action="store_true")
