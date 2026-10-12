import RoboVerify.Examples
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
