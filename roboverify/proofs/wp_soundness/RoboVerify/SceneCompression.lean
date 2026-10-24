import RoboVerify.FiniteLevels
import RoboVerify.Fragments

set_option autoImplicit false

namespace RoboVerify.SceneCompression
open FiniteLevels
variable {α : Type} (s : Scene α) (wf : s.WellFormed)

noncomputable def blockAt (a : α) (k : Nat) (h : k ≤ s.level a) : α :=
  (wf.complete a k h).choose

theorem at_spec (a : α) (k : Nat) (h : k ≤ s.level a) :
    s.root (blockAt s wf a k h) = s.root a ∧ s.level (blockAt s wf a k h) = k :=
  (wf.complete a k h).choose_spec

noncomputable def row (levels : List Nat) (a : α) : List α :=
  levels.filterMap fun k => if h : k ≤ s.level a then some (blockAt s wf a k h) else none

theorem mem_row (levels : List Nat) (a x : α) :
    x ∈ row s wf levels a ↔
      s.root x = s.root a ∧ s.level x ∈ levels ∧ s.level x ≤ s.level a := by
  classical
  simp only [row, List.mem_filterMap]
  constructor
  · rintro ⟨k, hk, hx⟩
    split at hx
    · rename_i h
      have he : blockAt s wf a k h = x := Option.some.inj hx
      have hs := at_spec s wf a k h
      rw [he] at hs
      exact ⟨hs.1, hs.2 ▸ hk, hs.2 ▸ h⟩
    · contradiction
  · rintro ⟨hr, hl, hh⟩
    refine ⟨s.level x, hl, ?_⟩
    simp only [dite_eq_left hh, Option.some.injEq]
    exact wf.unique _ x ((at_spec s wf a _ hh).1.trans hr.symm)
      (at_spec s wf a _ hh).2

theorem length_row (levels : List Nat) (a : α) :
    (row s wf levels a).length ≤ levels.length := List.length_filterMap_le _ _

noncomputable def retained (seeds : List α) (levels : List Nat) : List α :=
  seeds.flatMap (row s wf levels)

theorem mem_retained (seeds : List α) (levels : List Nat) (x : α) :
    x ∈ retained s wf seeds levels ↔
      ∃ a ∈ seeds, s.root x = s.root a ∧ s.level x ∈ levels ∧ s.level x ≤ s.level a := by
  simp only [retained, List.mem_flatMap, mem_row]

theorem length_retained (seeds : List α) (levels : List Nat) :
    (retained s wf seeds levels).length ≤ seeds.length * levels.length := by
  induction seeds with
  | nil => simp [retained]
  | cons a seeds ih =>
    have hr := length_row s wf levels a
    simp only [retained, List.flatMap_cons, List.length_append] at *
    simp only [List.length_cons, Nat.add_mul, Nat.one_mul]
    omega

def levels (seeds : List α) : List Nat := normalize (0 :: seeds.map s.level)

theorem zero_mem (seeds : List α) : 0 ∈ levels s seeds := by simp [levels]
theorem seed_level (seeds : List α) {a : α} (ha : a ∈ seeds) :
    s.level a ∈ levels s seeds := by
  simp only [levels, mem_normalize, List.mem_cons]
  exact Or.inr (List.mem_map.mpr ⟨a, ha, rfl⟩)
theorem levels_sorted (seeds : List α) : (levels s seeds).Pairwise (· < ·) :=
  sorted_normalize _
theorem levels_length (seeds : List α) : (levels s seeds).length ≤ seeds.length + 1 := by
  simpa [levels] using length_normalize (0 :: seeds.map s.level)

abbrev Domain (seeds : List α) := {x : α // x ∈ retained s wf seeds (levels s seeds)}

theorem seed_mem (seeds : List α) {a : α} (ha : a ∈ seeds) :
    a ∈ retained s wf seeds (levels s seeds) :=
  (mem_retained s wf seeds _ a).mpr ⟨a, ha, rfl, seed_level s seeds ha, Nat.le_refl _⟩

theorem level_mem (seeds : List α) (x : Domain s wf seeds) :
    s.level x.val ∈ levels s seeds := by
  obtain ⟨_, _, _, h, _⟩ := (mem_retained s wf seeds _ x.val).mp x.property
  exact h

theorem root_mem (seeds : List α) (x : Domain s wf seeds) :
    s.root x.val ∈ retained s wf seeds (levels s seeds) := by
  obtain ⟨a, ha, hr, _, _⟩ := (mem_retained s wf seeds _ x.val).mp x.property
  apply (mem_retained s wf seeds _ _).mpr
  refine ⟨a, ha, (wf.root_fixed _).trans hr, ?_, ?_⟩
  · rw [wf.root_zero]; exact zero_mem s seeds
  · rw [wf.root_zero]; exact Nat.zero_le _

noncomputable def compressed (seeds : List α) : Scene (Domain s wf seeds) where
  root x := ⟨s.root x.val, root_mem s wf seeds x⟩
  level x := (levels s seeds).idxOf (s.level x.val)

theorem compressed_wellFormed (seeds : List α) : (compressed s wf seeds).WellFormed := by
  let ls := levels s seeds
  have hs := levels_sorted s seeds
  constructor
  · intro x
    apply Subtype.ext
    exact wf.root_fixed x.val
  · intro x
    change ls.idxOf (s.level (s.root x.val)) = 0
    rw [wf.root_zero]
    exact rank_zero ls hs (zero_mem s seeds)
  · intro x y hr hl
    apply Subtype.ext
    apply wf.unique
    · exact congrArg Subtype.val hr
    · exact (rank_eq_iff ls hs (level_mem s wf seeds x) (level_mem s wf seeds y)).mp hl
  · intro x k hk
    have hl := List.idxOf_lt_length_of_mem (level_mem s wf seeds x)
    change ls.idxOf (s.level x.val) < ls.length at hl
    have hkl : k < ls.length := by change k ≤ ls.idxOf (s.level x.val) at hk; omega
    let h := ls[k]
    have hh : h ∈ ls := List.getElem_mem hkl
    have hr : ls.idxOf h = k := idxOf_get ls hs ⟨k, hkl⟩
    have hhx : h ≤ s.level x.val :=
      (rank_le_iff ls hs hh (level_mem s wf seeds x)).mp (hr ▸ hk)
    obtain ⟨y, hyroot, hylevel⟩ := wf.complete x.val h hhx
    have hym : y ∈ retained s wf seeds ls := by
      obtain ⟨a, ha, hxa, _, hla⟩ := (mem_retained s wf seeds _ x.val).mp x.property
      apply (mem_retained s wf seeds ls y).mpr
      exact ⟨a, ha, hyroot.trans hxa, hylevel ▸ hh, hylevel ▸ Nat.le_trans hhx hla⟩
    refine ⟨⟨y, hym⟩, ?_, ?_⟩
    · apply Subtype.ext; exact hyroot
    · change ls.idxOf (s.level y) = k
      rw [hylevel]; exact hr

theorem compressed_on (seeds : List α) (x y : Domain s wf seeds) :
    (compressed s wf seeds).On x y ↔ s.On x.val y.val := by
  simp only [Scene.On, compressed, Subtype.mk.injEq]
  rw [rank_le_iff _ (levels_sorted s seeds) (level_mem s wf seeds y) (level_mem s wf seeds x)]

theorem compressed_higher (seeds : List α) (x y : Domain s wf seeds) :
    (compressed s wf seeds).Higher x y ↔ s.Higher x.val y.val :=
  rank_le_iff _ (levels_sorted s seeds) (level_mem s wf seeds y) (level_mem s wf seeds x)

theorem compressed_scattered (seeds : List α) (x y : Domain s wf seeds) :
    (compressed s wf seeds).Scattered x y ↔ s.Scattered x.val y.val := by
  simp [Scene.Scattered, compressed]

theorem retained_bound (seeds : List α) :
    (retained s wf seeds (levels s seeds)).length ≤ seeds.length * (seeds.length + 1) :=
  Nat.le_trans (length_retained s wf seeds _) (Nat.mul_le_mul_left _ (levels_length s seeds))

theorem compressed_level_bound (seeds : List α) (x : Domain s wf seeds) :
    (compressed s wf seeds).level x ≤ seeds.length := by
  have h := List.idxOf_lt_length_of_mem (level_mem s wf seeds x)
  have hl := levels_length s seeds
  change (levels s seeds).idxOf (s.level x.val) ≤ seeds.length
  omega

end RoboVerify.SceneCompression
