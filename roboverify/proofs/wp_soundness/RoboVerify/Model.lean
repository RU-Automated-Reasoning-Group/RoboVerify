import Std

set_option autoImplicit false

namespace RoboVerify

/-- A coordinate-free representation of complete, separated support chains.
Root labels are physical objects; levels count direct-support edges. -/
structure Scene (α : Type) where
  root : α → α
  level : α → Nat

namespace Scene

variable {α : Type}

structure WellFormed (s : Scene α) : Prop where
  root_fixed : ∀ x, s.root (s.root x) = s.root x
  root_zero : ∀ x, s.level (s.root x) = 0
  unique : ∀ x y, s.root x = s.root y → s.level x = s.level y → x = y
  complete : ∀ x k, k ≤ s.level x →
    ∃ y, s.root y = s.root x ∧ s.level y = k

def On (s : Scene α) (x y : α) : Prop :=
  s.root x = s.root y ∧ s.level y ≤ s.level x

def Higher (s : Scene α) (x y : α) : Prop :=
  s.level y ≤ s.level x

def Scattered (s : Scene α) (x y : α) : Prop :=
  s.root x ≠ s.root y

def Clear (s : Scene α) (a : α) : Prop :=
  ∀ x, s.On x a → x = a

def Singleton (s : Scene α) (a : α) : Prop :=
  s.Clear a ∧ ∀ x, s.On a x → x = a

/-- This is direct support, not the proposed ON* rewrite. -/
def Direct (s : Scene α) (x y : α) : Prop :=
  s.root x = s.root y ∧ s.level x = s.level y + 1

/-- Independently defined reflexive transitive closure of direct support. -/
inductive Reach (s : Scene α) : α → α → Prop where
  | refl (x) : Reach s x x
  | step {x y z} : s.Direct x y → Reach s y z → Reach s x z

theorem on_refl (s : Scene α) (x : α) : s.On x x := ⟨rfl, Nat.le_refl _⟩

theorem on_trans {s : Scene α} {x y z : α}
    (hxy : s.On x y) (hyz : s.On y z) : s.On x z :=
  ⟨hxy.1.trans hyz.1, Nat.le_trans hyz.2 hxy.2⟩

theorem on_antisymm {s : Scene α} (wf : s.WellFormed) {x y : α}
    (hxy : s.On x y) (hyx : s.On y x) : x = y :=
  wf.unique x y hxy.1 (Nat.le_antisymm hyx.2 hxy.2)

theorem on_root {s : Scene α} (wf : s.WellFormed) (x : α) :
    s.On x (s.root x) := by
  exact ⟨(wf.root_fixed x).symm, by rw [wf.root_zero]; omega⟩

theorem root_ne_of_clear {s : Scene α} (wf : s.WellFormed)
    {a x : α} (ha : s.Clear a) (hxa : x ≠ a) : s.root x ≠ a := by
  intro hr
  have h := on_root wf x
  rw [hr] at h
  exact hxa (ha x h)

theorem into_clear {s : Scene α} {a x : α} (ha : s.Clear a) :
    s.On x a ↔ x = a := by
  constructor
  · exact ha x
  · rintro rfl
    exact on_refl s _

theorem from_singleton {s : Scene α} {a x : α} (ha : s.Singleton a) :
    s.On a x ↔ x = a := by
  constructor
  · exact ha.2 x
  · rintro rfl
    exact on_refl s _

theorem singleton_level_zero {s : Scene α} (wf : s.WellFormed)
    {a : α} (ha : s.Singleton a) : s.level a = 0 := by
  have hr := ha.2 (s.root a) (on_root wf a)
  simpa [hr] using wf.root_zero a

theorem level_le_clear {s : Scene α} {b x : α}
    (hb : s.Clear b) (hr : s.root x = s.root b) :
    s.level x ≤ s.level b := by
  by_cases h : s.level b ≤ s.level x
  · have hx := hb x ⟨hr, h⟩
    simp [hx]
  · omega

theorem clear_roots_ne {s : Scene α} {a b : α}
    (ha : s.Clear a) (hb : s.Clear b) (hne : a ≠ b) :
    s.root a ≠ s.root b := by
  intro hr
  have hab := level_le_clear hb hr
  exact hne ((ha b ⟨hr.symm, hab⟩).symm)

theorem reach_to_on {s : Scene α} {x y : α} (h : s.Reach x y) :
    s.On x y := by
  induction h with
  | refl x => exact on_refl s x
  | step h _ ih =>
    exact ⟨h.1.trans ih.1, by have := h.2; have := ih.2; omega⟩

theorem on_to_reach {s : Scene α} (wf : s.WellFormed) {x y : α}
    (h : s.On x y) : s.Reach x y := by
  by_cases heq : s.level x = s.level y
  · have hxy := wf.unique x y h.1 heq
    subst y
    exact .refl x
  · have hpos : 0 < s.level x := by have := h.2; omega
    obtain ⟨t, hroot, hlevel⟩ := wf.complete x (s.level x - 1) (by omega)
    have hxt : s.Direct x t := ⟨hroot.symm, by omega⟩
    have hty : s.On t y := ⟨hroot.trans h.1, by have := h.2; omega⟩
    exact .step hxt (on_to_reach wf hty)
termination_by s.level x
decreasing_by
  simp_wf
  omega

/-- The scene's ON* really is support reachability, for every block domain. -/
theorem on_iff_reach {s : Scene α} (wf : s.WellFormed) (x y : α) :
    s.On x y ↔ s.Reach x y :=
  ⟨on_to_reach wf, reach_to_on⟩

section Updates
variable [DecidableEq α]

/-- Execution changes the selected object's column and level, independently
of the logical rewrite formulas. -/
def put (s : Scene α) (a b : α) : Scene α where
  root x := if x = a then s.root b else s.root x
  level x := if x = a then s.level b + 1 else s.level x

def separate (s : Scene α) (a : α) : Scene α where
  root x := if x = a then a else s.root x
  level x := if x = a then 0 else s.level x

@[simp] theorem put_root_self (s : Scene α) (a b : α) :
    (s.put a b).root a = s.root b := by simp [put]
@[simp] theorem put_level_self (s : Scene α) (a b : α) :
    (s.put a b).level a = s.level b + 1 := by simp [put]
@[simp] theorem separate_root_self (s : Scene α) (a : α) :
    (s.separate a).root a = a := by simp [separate]
@[simp] theorem separate_level_self (s : Scene α) (a : α) :
    (s.separate a).level a = 0 := by simp [separate]

theorem on_put {s : Scene α} {a b : α}
    (ha : s.Clear a) (hb : s.Clear b) (x y : α) :
    (s.put a b).On x y ↔
      (x ≠ a ∧ s.On x y) ∨ (x = a ∧ (y = a ∨ s.On b y)) := by
  by_cases hx : x = a
  · subst x
    by_cases hy : y = a
    · subst y
      simp [On, put]
    · simp [On, put, hy]
      intro hr
      have := level_le_clear hb hr.symm
      omega
  · by_cases hy : y = a
    · subst y
      have hold : ¬ s.On x a := fun h => hx (ha x h)
      have hnew : ¬ (s.put a b).On x a := by
        simp [On, put, hx]
        intro hr
        have := level_le_clear hb hr
        omega
      simp [hx, hold, hnew]
    · simp [On, put, hx, hy]

theorem on_separate {s : Scene α} (wf : s.WellFormed) {a : α}
    (ha : s.Clear a) (x y : α) :
    (s.separate a).On x y ↔
      s.On x y ∧ (¬s.On x a ∨ s.On y a) := by
  simp only [into_clear ha]
  by_cases hx : x = a
  · subst x
    by_cases hy : y = a
    · subst y; simp [On, separate]
    · have hr := root_ne_of_clear wf ha hy
      simp [On, separate, hy, Ne.symm hr]
  · by_cases hy : y = a
    · subst y
      have hr := root_ne_of_clear wf ha hx
      simp only [into_clear ha]
      simp [separate, On, hx, hr]
    · simp [On, separate, hx, hy]

/-- The paper's additive rule requires the stronger singleton-source contract. -/
theorem on_attach {s : Scene α} {a b : α}
    (ha : s.Singleton a) (hb : s.Clear b) (x y : α) :
    (s.put a b).On x y ↔ s.On x y ∨ (s.On x a ∧ s.On b y) := by
  rw [on_put ha.1 hb, into_clear ha.1]
  by_cases hx : x = a
  · subst x
    simp [from_singleton ha]
  · simp [hx]

theorem separate_put (s : Scene α) {a b : α} (hne : a ≠ b) :
    (s.separate a).put a b = s.put a b := by
  cases s
  simp only [separate, put]
  simp [Ne.symm hne]
  constructor <;> funext x <;> split <;> simp_all

theorem separate_wellFormed {s : Scene α} (wf : s.WellFormed)
    {a : α} (ha : s.Clear a) : (s.separate a).WellFormed := by
  constructor
  · intro x
    by_cases hx : x = a
    · subst x; simp [separate]
    · have hr := root_ne_of_clear wf ha hx
      simp [separate, hx, hr, wf.root_fixed]
  · intro x
    by_cases hx : x = a
    · subst x; simp [separate]
    · have hr := root_ne_of_clear wf ha hx
      simp [separate, hx, hr, wf.root_zero]
  · intro x y hr hl
    by_cases hx : x = a <;> by_cases hy : y = a
    · exact hx.trans hy.symm
    · have hroot := root_ne_of_clear wf ha hy
      simp [separate, hx, hy, Ne.symm hroot] at hr
    · have hroot := root_ne_of_clear wf ha hx
      simp [separate, hx, hy, hroot] at hr
    · simp [separate, hx, hy] at hr hl
      exact wf.unique x y hr hl
  · intro x k hk
    by_cases hx : x = a
    · subst x
      have hk0 : k = 0 := by simpa using hk
      subst k
      exact ⟨a, rfl, by simp⟩
    · simp [separate, hx] at hk
      obtain ⟨y, hr, hl⟩ := wf.complete x k hk
      have hy : y ≠ a := by
        intro h; subst y
        exact hx (ha x ⟨hr.symm, by omega⟩)
      exact ⟨y, by simpa [separate, hx, hy] using hr,
        by simpa [separate, hy] using hl⟩

theorem put_wellFormed {s : Scene α} (wf : s.WellFormed)
    {a b : α} (ha : s.Clear a) (hb : s.Clear b) (hne : a ≠ b) :
    (s.put a b).WellFormed := by
  have hrb : s.root b ≠ a := root_ne_of_clear wf ha (Ne.symm hne)
  have hrab := clear_roots_ne ha hb hne
  constructor
  · intro x
    by_cases hx : x = a
    · subst x; simp [put, hrb, wf.root_fixed]
    · have hr := root_ne_of_clear wf ha hx
      simp [put, hx, hr, wf.root_fixed]
  · intro x
    by_cases hx : x = a
    · subst x; simp [put, hrb, wf.root_zero]
    · have hr := root_ne_of_clear wf ha hx
      simp [put, hx, hr, wf.root_zero]
  · intro x y hr hl
    by_cases hx : x = a <;> by_cases hy : y = a
    · exact hx.trans hy.symm
    · simp [put, hx, hy] at hr hl
      have := level_le_clear hb hr.symm
      omega
    · simp [put, hx, hy] at hr hl
      have := level_le_clear hb hr
      omega
    · simp [put, hx, hy] at hr hl
      exact wf.unique x y hr hl
  · intro x k hk
    by_cases hx : x = a
    · subst x
      simp only [put_level_self] at hk
      by_cases he : k = s.level b + 1
      · exact ⟨a, rfl, by simp [he]⟩
      · obtain ⟨y, hr, hl⟩ := wf.complete b k (by omega)
        have hy : y ≠ a := by intro h; subst y; exact hrab hr
        exact ⟨y, by simpa [put, hy] using hr,
          by simpa [put, hy] using hl⟩
    · simp [put, hx] at hk
      obtain ⟨y, hr, hl⟩ := wf.complete x k hk
      have hy : y ≠ a := by
        intro h; subst y
        exact hx (ha x ⟨hr.symm, by omega⟩)
      exact ⟨y, by simpa [put, hx, hy] using hr,
        by simpa [put, hy] using hl⟩

theorem separate_singleton {s : Scene α} (wf : s.WellFormed)
    {a : α} (ha : s.Clear a) : (s.separate a).Singleton a := by
  constructor
  · intro x h
    have := (on_separate wf ha x a).mp h
    exact ha x this.1
  · intro x h
    have hh := (on_separate wf ha a x).mp h
    rcases hh.2 with hn | hx
    · exact False.elim (hn (on_refl s a))
    · exact ha x hx

theorem separate_clear {s : Scene α} (wf : s.WellFormed)
    {a b : α} (ha : s.Clear a) (hb : s.Clear b) :
    (s.separate a).Clear b := by
  intro x h
  exact hb x ((on_separate wf ha x b).mp h).1

/-- An inhabited family of models: every object starts on its own at level zero. -/
def flat : Scene α := ⟨id, fun _ => 0⟩

omit [DecidableEq α] in
theorem flat_wellFormed : (flat : Scene α).WellFormed := by
  constructor
  · intro x; rfl
  · intro x; rfl
  · intro x y hr _; exact hr
  · intro x k hk
    exact ⟨x, rfl, by simpa [flat] using (Nat.eq_zero_of_le_zero hk).symm⟩

end Updates
end Scene
end RoboVerify
