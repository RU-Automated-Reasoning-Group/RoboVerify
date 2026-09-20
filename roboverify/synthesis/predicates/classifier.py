from synthesis.predicates.enumerate import enumerate_separator


def learn_classifier(positive, negative, scope, **kwargs):
    if not positive or not negative:
        from synthesis.predicates.enumerate import SearchResult

        return SearchResult("insufficient_examples")
    return enumerate_separator(
        [(s, True) for s in positive] + [(s, False) for s in negative],
        tuple(sorted(scope)),
        mode="classifier",
        **kwargs
    )


LearnClassifier = learn_classifier
