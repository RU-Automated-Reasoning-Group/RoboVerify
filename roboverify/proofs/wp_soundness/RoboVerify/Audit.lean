import RoboVerify.Examples
import RoboVerify.RelationalSummary
import RoboVerify.WitnessWP
import RoboVerify.FragmentSoundness
import RoboVerify.Placement
import RoboVerify.BlockProgram
import Lean

/- Audit every declaration in this development, including generated helpers.
A custom axiom, sorryAx, or native_decide axiom makes this command fail.
The whitelist is Lean's standard foundational axioms. -/
open Lean Elab Command in
run_elab do
  let env ← getEnv
  let allowed := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut checked : Nat := 0
  for (name, _) in env.constants.toList do
    if `RoboVerify |>.isPrefixOf name then
      for axiomName in ← Lean.collectAxioms name do
        unless allowed.contains axiomName do
          throwError "Unapproved axiom {axiomName} in {name}"
      checked := checked + 1
  if checked = 0 then
    throwError "Audit found no RoboVerify declarations"
  logInfo m!"Axiom audit passed for {checked} RoboVerify declarations."

#print axioms RoboVerify.Scene.on_iff_reach
#print axioms RoboVerify.Scene.direct_put
#print axioms RoboVerify.Formula.wpAttach_exact
#print axioms RoboVerify.Formula.wpPut_exact
#print axioms RoboVerify.Formula.wpPut_decomposition
#print axioms RoboVerify.Action.wp_exact
#print axioms RoboVerify.Command.verified_program
#print axioms RoboVerify.Command.loopFree_wp_exact
#print axioms RoboVerify.Command.chooseLoop_vc
#print axioms RoboVerify.Command.chooseIf_wp
#print axioms RoboVerify.WitnessWP.wpPut_exact
#print axioms RoboVerify.WitnessWP.wpSeparate_exact
#print axioms RoboVerify.WitnessWP.putAtom_qf
#print axioms RoboVerify.Fragments.lift_AF_action
#print axioms RoboVerify.Fragments.Program.vcgen_epr
#print axioms RoboVerify.Fragments.Program.vcgen_background_epr
#print axioms RoboVerify.Fragments.Program.verified_program
#print axioms RoboVerify.Fragments.Rules.sharedPut_exact
#print axioms RoboVerify.Fragments.EA.eval_close
#print axioms RoboVerify.Fragments.AE.eval_toFormula
#print axioms RoboVerify.Fragments.Placement.wpBlock_equivWitness
#print axioms RoboVerify.Fragments.Placement.wpTable_equivWitness
#print axioms RoboVerify.Fragments.Placement.wp_action_exact
#print axioms RoboVerify.Fragments.Placement.put_preserves_table
#print axioms RoboVerify.BlockProgram.Operation.wp_exact
#print axioms RoboVerify.BlockProgram.verified_program
#print axioms RoboVerify.BlockProgram.vcgen_epr
#print axioms RoboVerify.RelationalSummary.rewrite_correct
#print axioms RoboVerify.RelationalSummary.wp_exact
#print axioms RoboVerify.RelationalSummary.wp_weakest
#print axioms RoboVerify.RelationalSummary.implementation_refines
#print axioms RoboVerify.RelationalSummary.scattered_self_rejects
