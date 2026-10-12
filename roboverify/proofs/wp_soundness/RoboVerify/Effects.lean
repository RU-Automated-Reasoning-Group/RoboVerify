import RoboVerify.Model

set_option autoImplicit false

namespace RoboVerify.Scene
variable {α : Type} [DecidableEq α]

/-- Relational characterization of being at most one level above b.
The universal clause is necessary: an ordering alone cannot measure a gap. -/
def NextHeight (s : Scene α) (b c : α) : Prop :=
  s.Higher b c ∨ (s.Higher c b ∧
    ∀ t, (s.Higher c t ∧ ¬ s.Higher t c) → s.Higher b t)

omit [DecidableEq α] in
theorem nextHeight_iff {s : Scene α} (wf : s.WellFormed) (b c : α) :
    s.NextHeight b c ↔ s.level c ≤ s.level b + 1 := by
  unfold NextHeight Higher
  constructor
  · rintro (h | ⟨_, h⟩)
    · omega
    · by_cases hn : s.level c ≤ s.level b + 1
      · exact hn
      obtain ⟨t, _, ht⟩ := wf.complete c (s.level c - 1) (by omega)
      have hct : s.level t ≤ s.level c ∧ ¬ s.level c ≤ s.level t := by omega
      have := h t hct
      omega
  · intro h
    by_cases hbc : s.level c ≤ s.level b
    · exact Or.inl hbc
    · right
      constructor
      · omega
      · intro t ht; omega

def BottomHeight (s : Scene α) (c : α) : Prop := ∀ t, s.Higher t c

omit [DecidableEq α] in
theorem bottomHeight_iff {s : Scene α} (wf : s.WellFormed) (a c : α) :
    s.BottomHeight c ↔ s.level c = 0 := by
  constructor
  · intro h
    have hh := h (s.root a)
    unfold Higher at hh
    rw [wf.root_zero] at hh
    omega
  · intro h t
    simp [Higher, h]

/-- These are pre-state formulas, independently of execution. -/
def onPut (s : Scene α) (a b x y : α) : Prop :=
  (x ≠ a ∧ s.On x y) ∨ (x = a ∧ (y = a ∨ s.On b y))

def onSeparate (s : Scene α) (a x y : α) : Prop :=
  s.On x y ∧ (¬ s.On x a ∨ s.On y a)

def higherPut (s : Scene α) (a b x y : α) : Prop :=
  if x = a then (y = a ∨ s.NextHeight b y)
  else if y = a then (s.Higher x b ∧ ¬ s.Higher b x)
  else s.Higher x y

def higherSeparate (s : Scene α) (a x y : α) : Prop :=
  if x = a then (y = a ∨ s.BottomHeight y)
  else if y = a then True else s.Higher x y

def scatteredPut (s : Scene α) (a b x y : α) : Prop :=
  if x = a then (y ≠ a ∧ s.Scattered b y)
  else if y = a then s.Scattered x b else s.Scattered x y

def scatteredSeparate (s : Scene α) (a x y : α) : Prop :=
  if x = a then y ≠ a else if y = a then True else s.Scattered x y

theorem higher_put {s : Scene α} (wf : s.WellFormed) (a b x y : α) :
    (s.put a b).Higher x y ↔ s.higherPut a b x y := by
  unfold higherPut
  rw [nextHeight_iff wf]
  by_cases hx : x = a <;> by_cases hy : y = a <;>
    simp [Higher, put, hx, hy] <;> omega

theorem higher_separate {s : Scene α} (wf : s.WellFormed) (a x y : α) :
    (s.separate a).Higher x y ↔ s.higherSeparate a x y := by
  unfold higherSeparate
  rw [bottomHeight_iff wf a]
  by_cases hx : x = a <;> by_cases hy : y = a <;>
    simp [Higher, separate, hx, hy]

theorem scattered_put (s : Scene α) (a b x y : α) :
    (s.put a b).Scattered x y ↔ s.scatteredPut a b x y := by
  by_cases hx : x = a <;> by_cases hy : y = a <;>
    simp [scatteredPut, Scattered, put, hx, hy]

theorem scattered_separate {s : Scene α} (wf : s.WellFormed)
    {a : α} (ha : s.Clear a) (x y : α) :
    (s.separate a).Scattered x y ↔ s.scatteredSeparate a x y := by
  by_cases hx : x = a <;> by_cases hy : y = a
  · simp [scatteredSeparate, Scattered, separate, hx, hy]
  · have hr := root_ne_of_clear wf ha hy
    simp [scatteredSeparate, Scattered, separate, hx, hy, Ne.symm hr]
  · have hr := root_ne_of_clear wf ha hx
    simp [scatteredSeparate, Scattered, separate, hx, hy, hr]
  · simp [scatteredSeparate, Scattered, separate, hx, hy]

/-- Execution overwrites precisely the moved block's direct support edge. -/
theorem direct_put {s : Scene α} (wf : s.WellFormed) {a b : α}
    (ha : s.Clear a) (hb : s.Clear b) (hne : a ≠ b) (x y : α) :
    (s.put a b).Direct x y ↔
      (x = a ∧ y = b) ∨ (x ≠ a ∧ s.Direct x y) := by
  by_cases hx : x = a
  · subst x
    by_cases hy : y = a
    · subst y
      simp [Direct, put, hne]
    · simp [Direct, put, hy]
      constructor
      · intro h
        exact wf.unique y b h.1.symm (by omega)
      · intro h; subst y; simp
  · by_cases hy : y = a
    · subst y
      have hn : ¬ s.Direct x a := by
        intro h
        exact hx (ha x ⟨h.1, by have := h.2; omega⟩)
      simp only [hx, false_and, false_or, hn, and_false, iff_false]
      simp [Direct, put, hx]
      intro hr
      have := level_le_clear hb hr
      omega
    · simp [Direct, put, hx, hy]

theorem direct_separate {s : Scene α} (wf : s.WellFormed) {a : α}
    (ha : s.Clear a) (x y : α) :
    (s.separate a).Direct x y ↔ x ≠ a ∧ s.Direct x y := by
  by_cases hx : x = a
  · subst x
    simp [Direct, separate]
  · by_cases hy : y = a
    · subst y
      have hn : ¬ s.Direct x a := by
        intro h
        exact hx (ha x ⟨h.1, by have := h.2; omega⟩)
      have hr := root_ne_of_clear wf ha hx
      simp only [hn, and_false, iff_false]
      simp [Direct, separate, hx, hr]
    · simp [Direct, separate, hx, hy]

end RoboVerify.Scene
