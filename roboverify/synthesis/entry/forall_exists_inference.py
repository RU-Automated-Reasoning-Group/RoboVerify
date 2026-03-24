import synthesis.verification_lib.highlevel_verification_lib as highlevel_verification_lib
from synthesis.inference_lib import inference

if __name__ == "__main__":
    context = highlevel_verification_lib.HighLevelContext(mode="declare")
    inference.run_forall_exists_example(context)
    # inference.quant_enum_merge_test(context)
    # inference.run_promote_example(context)
