from synthesis.topdown import topdown

if __name__ == "__main__":
    all_inputs = topdown.run_reverse_hardcoded_dataset()
    topdown.topdown_synthesize(all_inputs, max_programs=100_000_000)
