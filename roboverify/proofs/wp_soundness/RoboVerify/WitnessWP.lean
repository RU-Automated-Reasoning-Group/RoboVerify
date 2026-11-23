import RoboVerify.HeightWitness
import RoboVerify.Formula

set_option autoImplicit false

namespace RoboVerify.WitnessWP
open Formula HeightWitness
variable {α : Type}

/-- These are macros over the existing vocabulary, not new relation symbols. -/
def nextRep (b u : α) : Formula α :=
  .conj (phys .higher u b) (.conj (.neg (phys .higher b u))
    (allPhysical fun t => imp
      (.conj (phys .higher t b) (.neg (phys .higher b t))) (phys .higher t u)))

def maximum (b : α) : Formula α := allPhysical fun t => phys .higher b t

@[simp] theorem eval_nextRep (s frozen : Scene α) (b u : α) :
    (nextRep b u).eval s frozen ↔ NextRep s b u := by
  simp only [nextRep, eval, eval_allPhysical, eval_imp, phys, relation, NextRep]
  simp only [and_imp]

@[simp] theorem eval_maximum (s frozen : Scene α) (b : α) :
    (maximum b).eval s frozen ↔ MaxHeight s b := by
  simp only [maximum, eval_allPhysical, phys, eval, relation, MaxHeight]

/-- Only the moved object's Higher row differs from the existing atom rule. -/
def putAtom (a b : α) (row : α → Formula α) (r : Relation)
    (x y : Object α) : Formula α :=
  match r, x, y with
  | .higher, some x, some y =>
    iteF (phys .eq x a) (.disj (phys .eq y a) (row y))
      (iteF (phys .eq y a)
        (.conj (phys .higher x b) (.neg (phys .higher b x))) (phys .higher x y))
  | _, _, _ => Formula.putAtom a b r x y

def separateAtom (a : α) (row : α → Formula α) (r : Relation)
    (x y : Object α) : Formula α :=
  match r, x, y with
  | .higher, some x, some y =>
    iteF (phys .eq x a) (.disj (phys .eq y a) (row y))
      (iteF (phys .eq y a) .truth (phys .higher x y))
  | _, _, _ => Formula.separateAtom a r x y

/-- Syntactic check for the local atom replacement, not the complete WP. -/
def QF : Formula α → Prop
  | .atom _ _ _ | .truth | .falsity => True
  | .neg p => QF p
  | .conj p q | .disj p q => QF p ∧ QF q
  | .all _ | .ex _ => False

theorem putAtom_qf (a b : α) (row : α → Formula α) (hq : ∀ x, QF (row x))
    (r : Relation) (x y : Object α) : QF (putAtom a b row r x y) := by
  cases r <;> cases x <;> cases y <;>
    simp [putAtom, Formula.putAtom, putPhysical, iteF, phys, QF, hq]

theorem separateAtom_qf (a : α) (row : α → Formula α) (hq : ∀ x, QF (row x))
    (r : Relation) (x y : Object α) : QF (separateAtom a row r x y) := by
  cases r <;> cases x <;> cases y <;>
    simp [separateAtom, Formula.separateAtom, separatePhysical, iteF, phys, QF, hq]

variable [DecidableEq α]

theorem rewrite_put {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    {a b : α} (ha : s.Clear a) (hb : s.Clear b) (row : α → Formula α)
    (hr : ∀ c, (row c).eval s frozen ↔ s.NextHeight b c) (q : Formula α) :
    (q.rewrite (putAtom a b row)).eval s frozen ↔ q.eval (s.put a b) frozen := by
  apply Formula.rewrite_correct
  intro r x y
  rw [← Formula.putAtom_correct wf frozen ha hb r x y]
  cases r <;> cases x <;> cases y <;>
    simp only [putAtom, Formula.putAtom, putPhysical, iteF, eval, eval_nextHeight, hr]

theorem rewrite_separate {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    {a : α} (ha : s.Clear a) (row : α → Formula α)
    (hr : ∀ c, (row c).eval s frozen ↔ s.BottomHeight c) (q : Formula α) :
    (q.rewrite (separateAtom a row)).eval s frozen ↔ q.eval (s.separate a) frozen := by
  apply Formula.rewrite_correct
  intro r x y
  rw [← Formula.separateAtom_correct wf frozen ha r x y]
  cases r <;> cases x <;> cases y <;>
    simp only [separateAtom, Formula.separateAtom, separatePhysical, iteF,
      eval, eval_bottomHeight, hr]

/-- Shared witnesses are scoped around the entire postcondition. Both guards
are universal formulas, and their occurrences here are antecedents. -/
def wpPut (a b : α) (q : Formula α) : Formula α :=
  .conj (.conj (clear a) (.conj (clear b) (.neg (phys .eq a b))))
    (.conj (imp (maximum b) (q.rewrite (putAtom a b (fun _ => .truth))))
      (allPhysical fun u => imp (nextRep b u)
        (q.rewrite (putAtom a b (fun c => phys .higher u c)))))

def wpSeparate (a : α) (q : Formula α) : Formula α :=
  .conj (clear a) (allPhysical fun u => imp (bottomHeight u)
    (q.rewrite (separateAtom a (fun c => phys .higher u c))))

theorem wpPut_exact {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (a b : α) (q : Formula α) :
    (wpPut a b q).eval s frozen ↔
      ∃ s', Formula.PutStep s a b s' ∧ q.eval s' frozen := by
  simp only [wpPut, eval, eval_clear, phys, relation, Option.some.injEq,
    eval_imp, eval_maximum, eval_allPhysical, eval_nextRep]
  have rep (u : α) (hu : NextRep s b u) (ha : s.Clear a) (hb : s.Clear b) :
      (q.rewrite (putAtom a b (fun c => phys .higher u c))).eval s frozen ↔
        q.eval (s.put a b) frozen := by
    apply rewrite_put wf frozen ha hb
    intro c
    rw [Scene.nextHeight_iff wf]
    simp only [phys, eval, relation, Scene.Higher, (nextRep_iff wf b u).mp hu]
  have max (hm : MaxHeight s b) (ha : s.Clear a) (hb : s.Clear b) :
      (q.rewrite (putAtom a b (fun _ => .truth))).eval s frozen ↔
        q.eval (s.put a b) frozen := by
    apply rewrite_put wf frozen ha hb
    intro c
    rw [Scene.nextHeight_iff wf]
    have hc := hm c
    unfold Scene.Higher at hc
    simp only [eval, true_iff]
    omega
  constructor
  · rintro ⟨⟨ha, hb, hab⟩, hm, hr⟩
    refine ⟨s.put a b, ⟨ha, hb, hab, rfl⟩, ?_⟩
    rcases coverage wf b with hmax | ⟨u, hu⟩
    · exact (max hmax ha hb).mp (hm hmax)
    · exact (rep u hu ha hb).mp (hr u hu)
  · rintro ⟨s', ⟨ha, hb, hab, rfl⟩, hq⟩
    exact ⟨⟨ha, hb, hab⟩, fun hm => (max hm ha hb).mpr hq,
      fun u hu => (rep u hu ha hb).mpr hq⟩

theorem wpSeparate_exact {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (a : α) (q : Formula α) :
    (wpSeparate a q).eval s frozen ↔
      ∃ s', Formula.SeparateStep s a s' ∧ q.eval s' frozen := by
  simp only [wpSeparate, eval, eval_clear, eval_allPhysical, eval_imp, eval_bottomHeight]
  have rep (u : α) (hu : s.BottomHeight u) (ha : s.Clear a) :
      (q.rewrite (separateAtom a (fun c => phys .higher u c))).eval s frozen ↔
        q.eval (s.separate a) frozen := by
    apply rewrite_separate wf frozen ha
    intro c
    rw [Scene.bottomHeight_iff wf a]
    have hz := (Scene.bottomHeight_iff wf a u).mp hu
    simp only [phys, eval, relation, Scene.Higher, hz]
    omega
  constructor
  · rintro ⟨ha, h⟩
    have hb : s.BottomHeight (s.root a) := by
      intro t; simp [Scene.Higher, wf.root_zero a]
    exact ⟨s.separate a, ⟨ha, rfl⟩, (rep _ hb ha).mp (h _ hb)⟩
  · rintro ⟨s', ⟨ha, rfl⟩, hq⟩
    exact ⟨ha, fun u hu => (rep u hu ha).mpr hq⟩

theorem wpPut_equiv {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (a b : α) (q : Formula α) :
    (wpPut a b q).eval s frozen ↔ (Formula.wpPut a b q).eval s frozen :=
  (wpPut_exact wf frozen a b q).trans (Formula.wpPut_exact wf frozen a b q).symm

end RoboVerify.WitnessWP
