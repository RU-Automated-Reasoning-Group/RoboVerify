"""Structural-search choice shared by CFG and standalone synthesis."""

SEARCH_STRATEGIES = ("mcmc", "enumeration")


def add_search_options(parser):
    parser.add_argument(
        "--search-strategy",
        choices=SEARCH_STRATEGIES,
        default="mcmc",
        help="Structural candidate generation: random MCMC mutations or deterministic enumeration",
    )


def validate_search_strategy(strategy):
    if strategy not in SEARCH_STRATEGIES:
        raise ValueError("Search strategy must be mcmc or enumeration")
    return strategy
