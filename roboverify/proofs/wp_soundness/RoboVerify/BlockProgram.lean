import RoboVerify.Placement
import RoboVerify.FragmentSoundness

set_option autoImplicit false

namespace RoboVerify.BlockProgram
open Fragments
variable {α V : Type}

/-- none is a literal table term; only some x denotes a mutable program name. -/
abbrev State (α V : Type) := RoboVerify.State α (Option V)

/-- Both the current and frozen entry scenes are complete separated towers.
The literal table is immutable and is represented by none in both syntaxes. -/
def Admissible (s : State α V) : Prop :=
  s.scene.WellFormed ∧ s.frozen.WellFormed ∧ s.names none = none

inductive Operation (V : Type) where
  | assign : V → Option V → Operation V
  | put : Option V → Option V → Operation V

namespace Operation
variable [DecidableEq V]

def rename (x : V) (value : Option V) (v : Option V) : Option V :=
  if v = some x then value else v

def wp : Operation V → AE (Option V) → AE (Option V)
  | .assign x value, q => q.map (rename x value)
  | .put a b, q => Placement.wp a b none q

instance : ActionWP (Option V) (Operation V) := ⟨wp⟩

def action : Operation V → Action α (Option V)
  | .assign x value => .assign (some x) (.name value)
  | .put a b => .put (.name a) (.name b)

variable [DecidableEq α]

def Step (op : Operation V) (s t : State α V) : Prop := op.action.Step s t

theorem wp_exact (op : Operation V) (s : State α V) (hs : Admissible s) (q : AE (Option V)) :
    (op.wp q).eval (relation s.scene s.frozen) s.names ↔
      (∃ t, op.Step s t) ∧ ∀ t, op.Step s t → q.eval (relation t.scene t.frozen) t.names := by
  cases op with
  | put a b => exact Placement.wp_action_exact s hs.1 a b none hs.2.2 q
  | assign x value =>
    have eq : s.names ∘ rename x value = (s.bind (some x) (s.names value)).names := by
      funext v
      simp only [Function.comp_def, rename, RoboVerify.State.bind]
      split <;> rfl
    simp only [wp, AE.eval_map, eq, Step, action, Action.Step, Term.eval]
    constructor
    · intro h
      exact ⟨⟨_, rfl⟩, fun t ht => ht ▸ h⟩
    · rintro ⟨_, h⟩
      exact h _ rfl

theorem preserves (op : Operation V) (s t : State α V) (hs : Admissible s)
    (run : op.Step s t) : Admissible t := by
  cases op with
  | assign x value =>
    cases run
    refine ⟨hs.1, hs.2.1, ?_⟩
    simpa [RoboVerify.State.bind] using hs.2.2
  | put a b =>
    change (Action.put (.name a) (.name b)).Step s t at run
    have ht := Placement.put_preserves_table a b none s t hs.2.2 run
    have wf := Action.step_wellFormed hs.1 run
    refine ⟨wf, ?_, ht⟩
    simp only [Action.Step, Term.eval] at run
    split at run
    · contradiction
    · obtain ⟨_, rfl⟩ := run; exact hs.2.1
    · obtain ⟨_, _, _, rfl⟩ := run; exact hs.2.1

end Operation

/-- Guards and specifications use the original relational vocabulary. The
binding map assigns program variables, and cannot assign the table literal. -/
inductive Command (V : Type) where
  | skip : Command V
  | action : Operation V → Command V
  | get : (b : Block) → QF (Option V ⊕ b.Index) → (V → Option V ⊕ b.Index) → Command V
  | seq : Command V → Command V → Command V
  | branch : (b : Block) → QF (Option V ⊕ b.Index) → (V → Option V ⊕ b.Index) →
      Command V → Command V → Command V
  | loop : (b : Block) → QF (Option V ⊕ b.Index) → (V → Option V ⊕ b.Index) →
      AF (Option V) → Command V → Command V

def bindMap {b : Block} (f : V → Option V ⊕ b.Index) : Option V → Option V ⊕ b.Index
  | none => .inl none
  | some v => f v

def Command.compile : Command V → Fragments.Program (Option V) (Operation V)
  | .skip => .skip
  | .action a => .action a
  | .get b g f => .get b g (bindMap f)
  | .seq c d => .seq c.compile d.compile
  | .branch b g f c d => .branch b g (bindMap f) c.compile d.compile
  | .loop b g f inv body => .loop b g (bindMap f) inv body.compile

theorem binding_preserves {b : Block} (f : V → Option V ⊕ b.Index) (s : State α V)
    (hs : Admissible s) (x : b.Index → Object α) :
    Admissible (Fragments.Program.bind s (bindMap f) x) :=
  ⟨hs.1, hs.2.1, hs.2.2⟩

variable [DecidableEq V] [DecidableEq α]

theorem actions_sound (c : Fragments.Program (Option V) (Operation V)) :
    Fragments.Program.ActionsSound (Operation.Step (α := α)) Admissible c := by
  induction c with
  | skip => trivial
  | get => trivial
  | action op =>
    intro s hs q h
    have exactWP := (op.wp_exact s hs q).mp h
    exact ⟨exactWP.1, fun t run => ⟨op.preserves s t hs run, exactWP.2 t run⟩⟩
  | seq c d hc hd => exact ⟨hc, hd⟩
  | branch b g f c d hc hd => exact ⟨hc, hd⟩
  | loop b g f inv body hb => exact hb

omit [DecidableEq V] [DecidableEq α] in
theorem bindings_sound (c : Command V) :
    Fragments.Program.BindingsSound (Admissible (α := α)) c.compile := by
  induction c with
  | skip => trivial
  | action => trivial
  | get b g f => exact binding_preserves f
  | seq c d hc hd => exact ⟨hc, hd⟩
  | branch b g f c d hc hd => exact ⟨binding_preserves f, hc, hd⟩
  | loop b g f inv body hb => exact ⟨binding_preserves f, hb⟩

def vcgen (p q : AF (Option V)) (c : Command V) : List (AE (Option V)) :=
  Fragments.Program.vcgen p c.compile q

/-- The action-soundness and binding obligations are discharged here, rather
than left as hypotheses of the concrete program theorem. -/
theorem verified_program (p q : AF (Option V)) (c : Command V)
    (valid : ∀ vc ∈ vcgen p q c, ∀ s : State α V, Admissible s →
      Fragments.Program.holds s vc) (s : State α V) (hs : Admissible s)
    (hp : p.eval (relation s.scene s.frozen) s.names) :
    (¬Fragments.Program.Fault Operation.Step c.compile s) ∧
      ∀ t, Fragments.Program.Exec Operation.Step c.compile s t →
        Admissible t ∧ q.eval (relation t.scene t.frozen) t.names :=
  Fragments.Program.verified_program p q c.compile (actions_sound c.compile)
    (bindings_sound c) valid s hs hp

omit [DecidableEq α] in
theorem vcgen_epr (p q : AF (Option V)) (c : Command V) :
    ∀ vc ∈ vcgen p q c, ∃ refutation : EA (Option V),
      ∀ (D : Type) (M : Relation → D → D → Prop) (ρ : Option V → D),
        refutation.eval M ρ ↔ ¬vc.eval M ρ :=
  Fragments.Program.vcgen_epr p q c.compile

end RoboVerify.BlockProgram
