import RoboVerify.Formula

set_option autoImplicit false

namespace RoboVerify

abbrev Names (α ν : Type) := ν → Object α
abbrev Assertion (α ν : Type) := Names α ν → Formula α

inductive Term (α ν : Type) where
  | literal : Object α → Term α ν
  | name : ν → Term α ν

def Term.eval {α ν : Type} (η : Names α ν) : Term α ν → Object α
  | .literal x => x
  | .name n => η n

structure State (α ν : Type) where
  scene : Scene α
  frozen : Scene α
  names : Names α ν

namespace State
variable {α ν : Type} [DecidableEq ν]

def bind (s : State α ν) (n : ν) (v : Object α) : State α ν :=
  { s with names := fun m => if m = n then v else s.names m }

def satisfies (s : State α ν) (q : Assertion α ν) : Prop :=
  (q s.names).eval s.scene s.frozen

def WellFormed (s : State α ν) : Prop := s.scene.WellFormed

end State

inductive Action (α ν : Type) where
  | assign : ν → Term α ν → Action α ν
  | put : Term α ν → Term α ν → Action α ν
  | get : ν → Assertion α ν → Action α ν

namespace Action
variable {α ν : Type} [DecidableEq α] [DecidableEq ν]

/-- Independent operational semantics. Put to the relational table separates
the source; table-as-source is illegal. Get may select any satisfying value. -/
def Step (act : Action α ν) (s t : State α ν) : Prop :=
  match act with
  | .assign n value => t = s.bind n (value.eval s.names)
  | .put a b =>
    match a.eval s.names, b.eval s.names with
    | none, _ => False
    | some a, none => s.scene.Clear a ∧ t = {s with scene := s.scene.separate a}
    | some a, some b =>
      s.scene.Clear a ∧ s.scene.Clear b ∧ a ≠ b ∧
        t = {s with scene := s.scene.put a b}
  | .get n condition =>
    ∃ v, (s.bind n v).satisfies condition ∧ t = s.bind n v

def wp (act : Action α ν) (q : Assertion α ν) : Assertion α ν :=
  fun η => match act with
  | .assign n value => q (fun m => if m = n then value.eval η else η m)
  | .put a b =>
    match a.eval η, b.eval η with
    | none, _ => .falsity
    | some a, none => Formula.wpSeparate a (q η)
    | some a, some b => Formula.wpPut a b (q η)
  | .get n condition =>
    .conj
      (.ex fun v => condition (fun m => if m = n then v else η m))
      (.all fun v => Formula.imp
        (condition (fun m => if m = n then v else η m))
        (q (fun m => if m = n then v else η m)))

theorem step_wellFormed {act : Action α ν} {s t : State α ν}
    (wf : s.WellFormed) (h : act.Step s t) : t.WellFormed := by
  cases act with
  | assign n value => cases h; exact wf
  | get n condition =>
    obtain ⟨v, _, rfl⟩ := h
    exact wf
  | put a b =>
    simp only [Step] at h
    split at h
    · contradiction
    · obtain ⟨ha, rfl⟩ := h
      exact Scene.separate_wellFormed wf ha
    · obtain ⟨ha, hb, hne, rfl⟩ := h
      exact Scene.put_wellFormed wf ha hb hne

/-- The action transformer is exact, including existence of a legal step
and every possible Get choice. It cannot prove an illegal action vacuously. -/
theorem wp_exact (act : Action α ν) (q : Assertion α ν) (s : State α ν)
    (wf : s.WellFormed) :
    s.satisfies (act.wp q) ↔
      (∃ t, act.Step s t) ∧ (∀ t, act.Step s t → t.satisfies q) := by
  classical
  cases act with
  | assign n value =>
    simp [wp, Step, State.satisfies, State.bind]
  | get n condition =>
    simp only [wp, State.satisfies, Formula.eval, Formula.eval_imp,
      Step, State.bind]
    constructor
    · rintro ⟨hex, hall⟩
      constructor
      · obtain ⟨v, hv⟩ := hex
        exact ⟨_, v, hv, rfl⟩
      · rintro t ⟨v, hv, rfl⟩
        exact hall v hv
    · rintro ⟨⟨t, v, hv, _⟩, hall⟩
      exact ⟨⟨v, hv⟩, fun w hw => hall _ ⟨w, hw, rfl⟩⟩
  | put a b =>
    simp only [wp, Step, State.satisfies]
    cases ha : a.eval s.names with
    | none => simp [Formula.eval]
    | some a' =>
      cases hb : b.eval s.names with
      | none =>
        simp only [Formula.wpSeparate_exact wf, Formula.SeparateStep]
        constructor
        · rintro ⟨s', ⟨hc, rfl⟩, hq⟩
          exact ⟨⟨_, hc, rfl⟩, by rintro t ⟨_, rfl⟩; exact hq⟩
        · rintro ⟨⟨t, hc, ht⟩, hall⟩
          exact ⟨s.scene.separate a', ⟨hc, rfl⟩, hall _ ⟨hc, rfl⟩⟩
      | some b' =>
        simp only [Formula.wpPut_exact wf, Formula.PutStep]
        constructor
        · rintro ⟨s', ⟨hc, hd, hne, rfl⟩, hq⟩
          exact ⟨⟨_, hc, hd, hne, rfl⟩, by rintro t ⟨_, _, _, rfl⟩; exact hq⟩
        · rintro ⟨⟨t, hc, hd, hne, ht⟩, hall⟩
          exact ⟨s.scene.put a' b', ⟨hc, hd, hne, rfl⟩,
            hall _ ⟨hc, hd, hne, rfl⟩⟩

end Action

/-- Loops carry a candidate invariant. No termination claim is encoded. -/
inductive Command (α ν : Type) where
  | skip : Command α ν
  | act : Action α ν → Command α ν
  | seq : Command α ν → Command α ν → Command α ν
  | loop : Assertion α ν → Assertion α ν → Command α ν → Command α ν

namespace Command
variable {α ν : Type} [DecidableEq α] [DecidableEq ν]

def wp (c : Command α ν) (q : Assertion α ν) : Assertion α ν :=
  match c with
  | .skip => q
  | .act a => a.wp q
  | .seq c d => c.wp (d.wp q)
  | .loop _ inv _ => inv

/-- VCs quantify over all well-formed model states and valuations. -/
def VC (c : Command α ν) (q : Assertion α ν) : Prop :=
  match c with
  | .skip => True
  | .act _ => True
  | .seq c d => c.VC (d.wp q) ∧ d.VC q
  | .loop guard inv body =>
    body.VC inv ∧
    (∀ s : State α ν, s.WellFormed → s.satisfies inv → s.satisfies guard →
      s.satisfies (body.wp inv)) ∧
    (∀ s : State α ν, s.WellFormed → s.satisfies inv → ¬s.satisfies guard →
      s.satisfies q)

inductive Exec : Command α ν → State α ν → State α ν → Prop where
  | skip (s) : Exec .skip s s
  | act {a s t} : a.Step s t → Exec (.act a) s t
  | seq {c d s u t} : Exec c s u → Exec d u t → Exec (.seq c d) s t
  | done {g i b s} : ¬s.satisfies g → Exec (.loop g i b) s s
  | more {g i b s u t} : s.satisfies g → Exec b s u →
      Exec (.loop g i b) u t → Exec (.loop g i b) s t

theorem exec_wellFormed {c : Command α ν} {s t : State α ν}
    (run : c.Exec s t) (wf : s.WellFormed) : t.WellFormed := by
  induction run with
  | skip => exact wf
  | act h => exact Action.step_wellFormed wf h
  | seq _ _ ihc ihd => exact ihd (ihc wf)
  | done => exact wf
  | more _ _ _ ihb ihloop => exact ihloop (ihb wf)

/-- Conditional partial correctness of the invariant-based VC generator. -/
theorem vc_sound {c : Command α ν} {s t : State α ν} {q : Assertion α ν}
    (run : c.Exec s t) (wf : s.WellFormed)
    (vc : c.VC q) (pre : s.satisfies (c.wp q)) : t.satisfies q := by
  induction run generalizing q with
  | skip => exact pre
  | act h => exact ((Action.wp_exact _ _ _ wf).mp pre).2 _ h
  | seq hc hd ihc ihd =>
    exact ihd (exec_wellFormed hc wf) vc.2 (ihc wf vc.1 pre)
  | done hg => exact vc.2.2 _ wf pre hg
  | more hg hb hl ihb ihl =>
    have inv := ihb wf vc.1 (vc.2.1 _ wf pre hg)
    exact ihl (exec_wellFormed hb wf) vc inv

/-- A finite execution prefix that reaches an illegal primitive or an empty
Get. Divergence is separate and is not considered a fault. -/
inductive Fault : Command α ν → State α ν → Prop where
  | act {a s} : (¬∃ t, a.Step s t) → Fault (.act a) s
  | seqLeft {c d s} : Fault c s → Fault (.seq c d) s
  | seqRight {c d s u} : Exec c s u → Fault d u → Fault (.seq c d) s
  | loopBody {g i b s} : s.satisfies g → Fault b s → Fault (.loop g i b) s
  | loopLater {g i b s u} : s.satisfies g → Exec b s u →
      Fault (.loop g i b) u → Fault (.loop g i b) s

theorem vc_no_fault {c : Command α ν} {s : State α ν} {q : Assertion α ν}
    (wf : s.WellFormed) (vc : c.VC q) (pre : s.satisfies (c.wp q)) :
    ¬c.Fault s := by
  intro fault
  induction fault generalizing q with
  | act hn => exact hn ((Action.wp_exact _ _ _ wf).mp pre).1
  | seqLeft _ ih => exact ih wf vc.1 pre
  | seqRight run _ ih =>
    exact ih (exec_wellFormed run wf) vc.2 (vc_sound run wf vc.1 pre)
  | loopBody hg _ ih => exact ih wf vc.1 (vc.2.1 _ wf pre hg)
  | loopLater hg run _ ih =>
    exact ih (exec_wellFormed run wf) vc
      (vc_sound run wf vc.1 (vc.2.1 _ wf pre hg))

/-- The main program theorem: every legal terminating execution satisfies Q,
and no finite execution prefix reaches a modeled fault. -/
theorem verified_program (c : Command α ν) (p q : Assertion α ν)
    (vc : c.VC q)
    (entry : ∀ s : State α ν, s.WellFormed → s.satisfies p →
      s.satisfies (c.wp q)) :
    ∀ s : State α ν, s.WellFormed → s.satisfies p →
      (¬ c.Fault s) ∧ (∀ t, c.Exec s t → t.WellFormed ∧ t.satisfies q) := by
  intro s wf hp
  have pre := entry s wf hp
  exact ⟨vc_no_fault wf vc pre,
    fun _ run => ⟨exec_wellFormed run wf, vc_sound run wf vc pre⟩⟩

/-- A syntactic restriction, used only for the exactness theorem below. -/
def LoopFree (c : Command α ν) : Prop :=
  match c with
  | .skip | .act _ => True
  | .seq c d => c.LoopFree ∧ d.LoopFree
  | .loop _ _ _ => False

omit [DecidableEq α] in
theorem loopFree_vc {c : Command α ν} (lf : c.LoopFree) (q : Assertion α ν) :
    c.VC q := by
  induction c generalizing q with
  | skip => trivial
  | act => trivial
  | seq c d hc hd => exact ⟨hc lf.1 _, hd lf.2 _⟩
  | loop => contradiction

theorem loopFree_wp_complete {c : Command α ν} (lf : c.LoopFree)
    {s : State α ν} (wf : s.WellFormed) (q : Assertion α ν)
    (safe : ¬c.Fault s) (post : ∀ t, c.Exec s t → t.satisfies q) :
    s.satisfies (c.wp q) := by
  classical
  induction c generalizing s q with
  | skip => exact post s (.skip s)
  | act a =>
    apply (Action.wp_exact a q s wf).mpr
    constructor
    · by_cases h : ∃ t, a.Step s t
      · exact h
      · exact False.elim (safe (.act h))
    · intro t h
      exact post t (.act h)
  | seq c d hc hd =>
    apply hc lf.1 wf (d.wp q)
    · intro h; exact safe (.seqLeft h)
    · intro u run
      apply hd lf.2 (exec_wellFormed run wf) q
      · intro h; exact safe (.seqRight run h)
      · intro t tail; exact post t (.seq run tail)
  | loop => contradiction

/-- For loop-free programs the calculation is weakest for fault-free,
all-outcome correctness. Loop invariants yield only a sufficient condition. -/
theorem loopFree_wp_exact {c : Command α ν} (lf : c.LoopFree)
    (s : State α ν) (wf : s.WellFormed) (q : Assertion α ν) :
    s.satisfies (c.wp q) ↔
      (¬c.Fault s) ∧ (∀ t, c.Exec s t → t.satisfies q) := by
  constructor
  · intro h
    exact ⟨vc_no_fault wf (loopFree_vc lf q) h,
      fun _ run => vc_sound run wf (loopFree_vc lf q) h⟩
  · rintro ⟨hsafe, hpost⟩
    exact loopFree_wp_complete lf wf q hsafe hpost

theorem loopFree_terminates {c : Command α ν} (lf : c.LoopFree)
    {s : State α ν} (wf : s.WellFormed) (q : Assertion α ν)
    (pre : s.satisfies (c.wp q)) : ∃ t, c.Exec s t := by
  induction c generalizing s q with
  | skip => exact ⟨s, .skip s⟩
  | act a =>
    obtain ⟨t, ht⟩ := ((Action.wp_exact a q s wf).mp pre).1
    exact ⟨t, .act ht⟩
  | seq c d hc hd =>
    obtain ⟨u, run⟩ := hc lf.1 wf (d.wp q) pre
    have mid := vc_sound run wf (loopFree_vc lf.1 _) pre
    obtain ⟨t, tail⟩ := hd lf.2 (exec_wellFormed run wf) q mid
    exact ⟨t, .seq run tail⟩
  | loop => contradiction

/-- An existential guard is implemented by testing existence and then Get.
Binding stability is the precise obligation often expressed in prose as
"the invariant does not mention the local witness name". -/
def existsGuard (n : ν) (g : Assertion α ν) : Assertion α ν :=
  fun η => .ex fun v => g (fun m => if m = n then v else η m)

def chooseLoop (n : ν) (g inv : Assertion α ν) (body : Command α ν) :
    Command α ν :=
  .loop (existsGuard n g) inv (.seq (.act (.get n g)) body)

omit [DecidableEq α] in
theorem chooseLoop_vc (n : ν) (g inv q : Assertion α ν) (body : Command α ν)
    (bodyVC : body.VC inv)
    (stable : ∀ (s : State α ν) v,
      s.satisfies inv → (s.bind n v).satisfies inv)
    (preserve : ∀ s : State α ν, s.WellFormed → s.satisfies inv →
      s.satisfies g → s.satisfies (body.wp inv))
    (exit : ∀ s : State α ν, s.WellFormed → s.satisfies inv →
      ¬s.satisfies (existsGuard n g) → s.satisfies q) :
    (chooseLoop n g inv body).VC q := by
  refine ⟨⟨True.intro, bodyVC⟩, ?_, exit⟩
  intro s wf hi hg
  change ((Formula.conj
    (.ex fun v => g (fun m => if m = n then v else s.names m))
    (.all fun v => Formula.imp
      (g (fun m => if m = n then v else s.names m))
      ((body.wp inv) (fun m => if m = n then v else s.names m)))).eval
        s.scene s.frozen)
  simp only [Formula.eval, Formula.eval_imp]
  refine ⟨hg, ?_⟩
  intro v hv
  exact preserve (s.bind n v) wf (stable s v hi) hv

end Command
end RoboVerify
