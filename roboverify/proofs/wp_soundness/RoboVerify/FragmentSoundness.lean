import RoboVerify.FragmentVC
import RoboVerify.Program

set_option autoImplicit false

namespace RoboVerify.Fragments
namespace Program
variable {α V : Type}

/-- Interpret the deep relational syntax in the existing RoboVerify states. -/
def holds (s : State α V) (q : AE V) : Prop :=
  q.eval (relation s.scene s.frozen) s.names

def guard (s : State α V) {b : Block} (g : QF (V ⊕ b.Index))
    (x : b.Index → Object α) : Prop :=
  g.eval (relation s.scene s.frozen) (Sum.elim s.names x)

def bind (s : State α V) {b : Block} (f : V → V ⊕ b.Index)
    (x : b.Index → Object α) : State α V :=
  {s with names := Sum.elim s.names x ∘ f}

abbrev AtomicStep (α V : Type) :=
  (AE V → AE V) → State α V → State α V → Prop

/-- Only actions actually present in the program need this contract. Atomic
WP soundness includes existence of a successor and preservation of the scene
domain. The theorem does not assume every function AE -> AE is a sound WP. -/
def ActionsSound (step : AtomicStep α V) : Program V → Prop
  | .skip | .get _ _ _ => True
  | .action transform => ∀ s, s.scene.WellFormed → ∀ q,
      holds s (transform q) → (∃ t, step transform s t) ∧
        ∀ t, step transform s t → t.scene.WellFormed ∧ holds t q
  | .seq c d | .branch _ _ _ c d => ActionsSound step c ∧ ActionsSound step d
  | .loop _ _ _ _ body => ActionsSound step body

def ValidVCs (c : Program V) (q : AE V) : Prop :=
  ∀ vc ∈ c.vcs q, ∀ s : State α V, s.scene.WellFormed → holds s vc

theorem valid_seq (c d : Program V) (q : AE V) :
    ValidVCs (α := α) (.seq c d) q ↔
      ValidVCs (α := α) c (d.wp q) ∧ ValidVCs (α := α) d q := by
  simp [ValidVCs, vcs, or_imp, forall_and]

theorem valid_branch (b : Block) (g : QF (V ⊕ b.Index)) (f : V → V ⊕ b.Index)
    (c d : Program V) (q : AE V) :
    ValidVCs (α := α) (.branch b g f c d) q ↔
      ValidVCs (α := α) c q ∧ ValidVCs (α := α) d q := by
  simp [ValidVCs, vcs, or_imp, forall_and]

theorem valid_loop (b : Block) (g : QF (V ⊕ b.Index)) (f : V → V ⊕ b.Index)
    (inv : AF V) (body : Program V) (q : AE V) :
    ValidVCs (α := α) (.loop b g f inv body) q ↔
      (∀ s : State α V, s.scene.WellFormed →
        holds s (Rules.preserve b g f inv (body.wp inv.toAE))) ∧
      (∀ s : State α V, s.scene.WellFormed → holds s (Rules.exit b g inv q)) ∧
      ValidVCs (α := α) body inv.toAE := by
  simp [ValidVCs, vcs, or_imp, forall_and]

/-- Independent operational semantics with tuple-valued demonic selection. -/
inductive Exec (step : AtomicStep α V) : Program V → State α V → State α V → Prop where
  | skip (s) : Exec step .skip s s
  | action {a s t} : step a s t → Exec step (.action a) s t
  | get {b g f s} (x) : guard s g x → Exec step (.get b g f) s (bind s f x)
  | seq {c d s u t} : Exec step c s u → Exec step d u t → Exec step (.seq c d) s t
  | yes {b g f c d s t} (x) : guard s g x → Exec step c (bind s f x) t →
      Exec step (.branch b g f c d) s t
  | no {b g f c d s t} : (∀ x, ¬guard s g x) → Exec step d s t →
      Exec step (.branch b g f c d) s t
  | done {b g f inv body s} : (∀ x, ¬guard s g x) →
      Exec step (.loop b g f inv body) s s
  | more {b g f inv body s u t} (x) : guard s g x →
      Exec step body (bind s f x) u → Exec step (.loop b g f inv body) u t →
      Exec step (.loop b g f inv body) s t

theorem vc_sound {step : AtomicStep α V} {c : Program V} {s t : State α V}
    (run : Exec step c s t) {q : AE V} (actions : ActionsSound step c)
    (vc : ValidVCs (α := α) c q) (wf : s.scene.WellFormed) (pre : holds s (c.wp q)) :
    t.scene.WellFormed ∧ holds t q := by
  induction run generalizing q with
  | skip => exact ⟨wf, pre⟩
  | action h => exact (actions _ wf _ pre).2 _ h
  | get x hg =>
    rw [wp, holds, Rules.eval_get] at pre
    exact ⟨wf, pre.2 x hg⟩
  | seq hc hd ihc ihd =>
    have vcs := (valid_seq _ _ _).mp vc
    have mid := ihc actions.1 vcs.1 wf pre
    exact ihd actions.2 vcs.2 mid.1 mid.2
  | yes x hg run ih =>
    have vcs := (valid_branch _ _ _ _ _ _).mp vc
    rw [wp, holds, Rules.eval_chooseIf] at pre
    exact ih actions.1 vcs.1 wf (pre.1 x hg)
  | no hg run ih =>
    have vcs := (valid_branch _ _ _ _ _ _).mp vc
    rw [wp, holds, Rules.eval_chooseIf] at pre
    exact ih actions.2 vcs.2 wf (pre.2 hg)
  | done hg =>
    have vcs := (valid_loop _ _ _ _ _ _).mp vc
    have he := vcs.2.1 _ wf
    rw [holds, Rules.eval_exit] at he
    rw [wp, holds, AF.eval_toAE] at pre
    exact ⟨wf, he ⟨pre, hg⟩⟩
  | more x hg body tail ihb ihl =>
    have vcs := (valid_loop _ _ _ _ _ _).mp vc
    have hp := vcs.1 _ wf
    rw [holds, Rules.eval_preserve] at hp
    rw [wp, holds, AF.eval_toAE] at pre
    have mid := ihb actions vcs.2.2 wf (hp x ⟨pre, hg⟩)
    exact ihl actions vc mid.1 mid.2

inductive Fault (step : AtomicStep α V) : Program V → State α V → Prop where
  | action {a s} : (¬∃ t, step a s t) → Fault step (.action a) s
  | get {b g f s} : (∀ x, ¬guard s g x) → Fault step (.get b g f) s
  | seqLeft {c d s} : Fault step c s → Fault step (.seq c d) s
  | seqRight {c d s u} : Exec step c s u → Fault step d u → Fault step (.seq c d) s
  | yes {b g f c d s} (x) : guard s g x → Fault step c (bind s f x) →
      Fault step (.branch b g f c d) s
  | no {b g f c d s} : (∀ x, ¬guard s g x) → Fault step d s →
      Fault step (.branch b g f c d) s
  | loopBody {b g f inv body s} (x) : guard s g x → Fault step body (bind s f x) →
      Fault step (.loop b g f inv body) s
  | loopLater {b g f inv body s u} (x) : guard s g x →
      Exec step body (bind s f x) u → Fault step (.loop b g f inv body) u →
      Fault step (.loop b g f inv body) s

theorem vc_no_fault {step : AtomicStep α V} {c : Program V} {s : State α V}
    {q : AE V} (actions : ActionsSound step c) (vc : ValidVCs (α := α) c q)
    (wf : s.scene.WellFormed) (pre : holds s (c.wp q)) : ¬Fault step c s := by
  intro fault
  induction fault generalizing q with
  | action hn => exact hn (actions _ wf _ pre).1
  | get hg =>
    rw [wp, holds, Rules.eval_get] at pre
    obtain ⟨x, hx⟩ := pre.1
    exact hg x hx
  | seqLeft _ ih => exact ih actions.1 ((valid_seq _ _ _).mp vc).1 wf pre
  | seqRight run _ ih =>
    have vcs := (valid_seq _ _ _).mp vc
    have mid := vc_sound run actions.1 vcs.1 wf pre
    exact ih actions.2 vcs.2 mid.1 mid.2
  | yes x hg _ ih =>
    rw [wp, holds, Rules.eval_chooseIf] at pre
    exact ih actions.1 ((valid_branch _ _ _ _ _ _).mp vc).1 wf (pre.1 x hg)
  | no hg _ ih =>
    rw [wp, holds, Rules.eval_chooseIf] at pre
    exact ih actions.2 ((valid_branch _ _ _ _ _ _).mp vc).2 wf (pre.2 hg)
  | loopBody x hg _ ih =>
    have vcs := (valid_loop _ _ _ _ _ _).mp vc
    have hp := vcs.1 _ wf
    rw [holds, Rules.eval_preserve] at hp
    rw [wp, holds, AF.eval_toAE] at pre
    exact ih actions vcs.2.2 wf (hp x ⟨pre, hg⟩)
  | loopLater x hg run _ ih =>
    have vcs := (valid_loop _ _ _ _ _ _).mp vc
    have hp := vcs.1 _ wf
    rw [holds, Rules.eval_preserve] at hp
    rw [wp, holds, AF.eval_toAE] at pre
    have mid := vc_sound run actions vcs.2.2 wf (hp x ⟨pre, hg⟩)
    exact ih actions vc mid.1 mid.2

/-- Program soundness for the restricted, normalizing generator interpreted
over RoboVerify's scenes. Termination is not asserted. -/
theorem verified_program {step : AtomicStep α V} (p q : AF V) (c : Program V)
    (actions : ActionsSound step c)
    (valid : ∀ vc ∈ vcgen p c q, ∀ s : State α V,
      s.scene.WellFormed → holds s vc) (s : State α V) (wf : s.scene.WellFormed)
    (hp : p.eval (relation s.scene s.frozen) s.names) :
    (¬Fault step c s) ∧ ∀ t, Exec step c s t →
      t.scene.WellFormed ∧ q.eval (relation t.scene t.frozen) t.names := by
  have entry := valid _ (List.mem_cons_self) s wf
  rw [holds, AF.eval_implies] at entry
  have vc : ValidVCs c q.toAE := fun v hv => valid v (List.mem_cons_of_mem _ hv)
  refine ⟨vc_no_fault actions vc wf (entry hp), ?_⟩
  intro t run
  have post := vc_sound run actions vc wf (entry hp)
  exact ⟨post.1, (AF.eval_toAE _ _ _).mp post.2⟩

end Program
end RoboVerify.Fragments
