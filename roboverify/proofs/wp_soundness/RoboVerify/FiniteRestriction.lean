import RoboVerify.FiniteLevels

set_option autoImplicit false

namespace RoboVerify.FiniteRestriction
open FiniteLevels
variable {α β : Type} (s : Scene α) (wf : s.WellFormed)
  (embed : β → α) (injective : ∀ x y, embed x = embed y → x = y)
  (objects : List β) (covers : ∀ x, x ∈ objects)

/-- Restricting ON* to finitely many objects leaves finite chains. Ranking each
chain separately reconstructs a well-formed frozen scene on exactly those objects. -/
noncomputable def column (x : β) : List Nat := by
  classical
  exact normalize ((objects.filter fun y => decide (s.root (embed y) = s.root (embed x))).map
    fun y => s.level (embed y))

include covers in
theorem mem_column (x : β) (k : Nat) :
    k ∈ column s embed objects x ↔
      ∃ y, s.root (embed y) = s.root (embed x) ∧ s.level (embed y) = k := by
  classical
  simp only [column, mem_normalize, List.mem_map, List.mem_filter, decide_eq_true_eq]
  constructor
  · rintro ⟨y, ⟨_, hr⟩, hl⟩; exact ⟨y, hr, hl⟩
  · rintro ⟨y, hr, hl⟩; exact ⟨y, ⟨covers y, hr⟩, hl⟩

theorem column_sorted (x : β) : (column s embed objects x).Pairwise (· < ·) :=
  sorted_normalize _

include covers in
theorem self_mem (x : β) : s.level (embed x) ∈ column s embed objects x :=
  (mem_column s embed objects covers x _).mpr ⟨x, rfl, rfl⟩

theorem column_eq {x y : β} (h : s.root (embed x) = s.root (embed y)) :
    column s embed objects x = column s embed objects y := by
  classical
  simp only [column, h]

noncomputable def chooseLevel (x : β) (k : Nat) (hk : k ∈ column s embed objects x) : β :=
  ((mem_column s embed objects covers x k).mp hk).choose

theorem chooseLevel_spec (x : β) (k : Nat) (hk : k ∈ column s embed objects x) :
    s.root (embed (chooseLevel s embed objects covers x k hk)) = s.root (embed x) ∧
      s.level (embed (chooseLevel s embed objects covers x k hk)) = k :=
  ((mem_column s embed objects covers x k).mp hk).choose_spec

include covers in
theorem column_nonempty (x : β) : 0 < (column s embed objects x).length := by
  have := List.idxOf_lt_length_of_mem (self_mem s embed objects covers x)
  omega

noncomputable def root (x : β) : β :=
  chooseLevel s embed objects covers x
    ((column s embed objects x)[0]'(column_nonempty s embed objects covers x))
    (List.getElem_mem (column_nonempty s embed objects covers x))

theorem root_spec (x : β) :
    s.root (embed (root s embed objects covers x)) = s.root (embed x) ∧
      s.level (embed (root s embed objects covers x)) =
        (column s embed objects x)[0]'(column_nonempty s embed objects covers x) :=
  chooseLevel_spec s embed objects covers x _ _

include wf injective in
theorem root_eq_iff (x y : β) :
    root s embed objects covers x = root s embed objects covers y ↔
      s.root (embed x) = s.root (embed y) := by
  constructor
  · intro h
    exact (root_spec s embed objects covers x).1.symm.trans
      ((congrArg (fun z => s.root (embed z)) h).trans (root_spec s embed objects covers y).1)
  · intro h
    apply injective
    apply wf.unique
    · exact ((root_spec s embed objects covers x).1.trans h).trans
        (root_spec s embed objects covers y).1.symm
    · rw [(root_spec s embed objects covers x).2, (root_spec s embed objects covers y).2]
      exact getElem_congr_coll (column_eq s embed objects h)

noncomputable def scene : Scene β where
  root := root s embed objects covers
  level x := (column s embed objects x).idxOf (s.level (embed x))

include wf injective in
theorem wellFormed : (scene s embed objects covers).WellFormed := by
  constructor
  · intro x
    apply (root_eq_iff s wf embed injective objects covers _ x).mpr
    exact (root_spec s embed objects covers x).1
  · intro x
    change (column s embed objects (root s embed objects covers x)).idxOf
      (s.level (embed (root s embed objects covers x))) = 0
    rw [column_eq s embed objects (root_spec s embed objects covers x).1,
      (root_spec s embed objects covers x).2]
    exact idxOf_get _ (column_sorted s embed objects x)
      ⟨0, column_nonempty s embed objects covers x⟩
  · intro x y hr hl
    have he := (root_eq_iff s wf embed injective objects covers x y).mp hr
    have hc := column_eq s embed objects he
    apply injective
    apply wf.unique _ _ he
    change (column s embed objects x).idxOf (s.level (embed x)) =
      (column s embed objects y).idxOf (s.level (embed y)) at hl
    rw [← hc] at hl
    apply (rank_eq_iff _ (column_sorted s embed objects x)
      (self_mem s embed objects covers x) ?_).mp hl
    rw [hc]; exact self_mem s embed objects covers y
  · intro x k hk
    let ls := column s embed objects x
    have hx := List.idxOf_lt_length_of_mem (self_mem s embed objects covers x)
    change ls.idxOf (s.level (embed x)) < ls.length at hx
    change k ≤ ls.idxOf (s.level (embed x)) at hk
    have hkl : k < ls.length := by omega
    have hm : ls[k] ∈ ls := List.getElem_mem hkl
    let y := chooseLevel s embed objects covers x ls[k] hm
    have hy := chooseLevel_spec s embed objects covers x ls[k] hm
    refine ⟨y, (root_eq_iff s wf embed injective objects covers y x).mpr hy.1, ?_⟩
    change (column s embed objects y).idxOf (s.level (embed y)) = k
    rw [column_eq s embed objects hy.1, hy.2]
    exact idxOf_get ls (column_sorted s embed objects x) ⟨k, hkl⟩

include wf injective in
theorem on_exact (x y : β) :
    (scene s embed objects covers).On x y ↔ s.On (embed x) (embed y) := by
  simp only [Scene.On, scene, root_eq_iff s wf embed injective objects covers]
  constructor
  · rintro ⟨hr, hl⟩
    refine ⟨hr, ?_⟩
    rw [column_eq s embed objects hr] at hl
    apply (rank_le_iff _ (column_sorted s embed objects y)
      (self_mem s embed objects covers y) ?_).mp hl
    rw [← column_eq s embed objects hr]; exact self_mem s embed objects covers x
  · rintro ⟨hr, hl⟩
    refine ⟨hr, ?_⟩
    rw [column_eq s embed objects hr]
    apply (rank_le_iff _ (column_sorted s embed objects y)
      (self_mem s embed objects covers y) ?_).mpr hl
    rw [← column_eq s embed objects hr]; exact self_mem s embed objects covers x

theorem column_length (x : β) : (column s embed objects x).length ≤ objects.length := by
  classical
  apply Nat.le_trans (length_normalize _)
  simp only [List.length_map]
  exact List.length_filter_le _ _

theorem level_bound (x : β) : (scene s embed objects covers).level x < objects.length := by
  have h := List.idxOf_lt_length_of_mem (self_mem s embed objects covers x)
  exact Nat.lt_of_lt_of_le h (column_length s embed objects x)

end RoboVerify.FiniteRestriction
