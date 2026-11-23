import RoboVerify.Effects

set_option autoImplicit false
set_option warningAsError true

namespace RoboVerify.HeightWitness
variable {α : Type}

/-- Abbreviations for formulas over the existing Higher relation, not new
symbols in the assertion vocabulary. All variables here are physical blocks. -/
def NextRep (s : Scene α) (b u : α) : Prop :=
  s.Higher u b ∧ ¬s.Higher b u ∧
    ∀ t, s.Higher t b → ¬s.Higher b t → s.Higher t u

def MaxHeight (s : Scene α) (b : α) : Prop := ∀ t, s.Higher b t

theorem nextRep_iff {s : Scene α} (wf : s.WellFormed) (b u : α) :
    NextRep s b u ↔ s.level u = s.level b + 1 := by
  unfold NextRep Scene.Higher
  constructor
  · rintro ⟨hge, hnot, hmin⟩
    obtain ⟨t, _, ht⟩ := wf.complete u (s.level b + 1) (by omega)
    have hm := hmin t (by omega) (by omega)
    omega
  · intro hu
    refine ⟨by omega, by omega, ?_⟩
    intro t hge hnot
    omega

theorem coverage {s : Scene α} (wf : s.WellFormed) (b : α) :
    MaxHeight s b ∨ ∃ u, NextRep s b u := by
  classical
  by_cases hm : MaxHeight s b
  · exact Or.inl hm
  · right
    change ¬ (∀ t, s.Higher b t) at hm
    obtain ⟨c, hc⟩ := Classical.not_forall.mp hm
    obtain ⟨u, _, hu⟩ := wf.complete c (s.level b + 1)
      (by unfold Scene.Higher at hc; omega)
    exact ⟨u, (nextRep_iff wf b u).mpr hu⟩


end RoboVerify.HeightWitness
