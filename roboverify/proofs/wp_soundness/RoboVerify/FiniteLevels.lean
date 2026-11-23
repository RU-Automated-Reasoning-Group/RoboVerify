import RoboVerify.Model

set_option autoImplicit false

namespace RoboVerify.FiniteLevels

/-- A finite set of natural heights, represented in increasing order. This
construction compresses gaps without imposing a bound on the original heights. -/
def insert (n : Nat) : List Nat → List Nat
  | [] => [n]
  | a :: xs => if n < a then n :: a :: xs else
      if n = a then a :: xs else a :: insert n xs

theorem mem_insert (n x : Nat) (xs : List Nat) :
    x ∈ insert n xs ↔ x = n ∨ x ∈ xs := by
  induction xs with
  | nil => simp [insert]
  | cons a xs ih =>
    simp only [insert]
    split
    · simp
    · split
      · simp_all
      · simp only [List.mem_cons, ih]
        exact or_left_comm

theorem length_insert (n : Nat) (xs : List Nat) :
    (insert n xs).length ≤ xs.length + 1 := by
  induction xs with
  | nil => simp [insert]
  | cons a xs ih =>
    simp only [insert]
    split <;> (try split) <;> simp_all only [List.length_cons] <;> omega

theorem sorted_insert (n : Nat) (xs : List Nat) (h : xs.Pairwise (· < ·)) :
    (insert n xs).Pairwise (· < ·) := by
  induction xs with
  | nil => simp [insert]
  | cons a xs ih =>
    obtain ⟨ha, hs⟩ := List.pairwise_cons.mp h
    simp only [insert]
    split
    · rename_i hn
      apply List.pairwise_cons.mpr
      exact ⟨fun x hx => by rcases List.mem_cons.mp hx with rfl | hx
                            · exact hn
                            · exact Nat.lt_trans hn (ha x hx), h⟩
    · rename_i hn
      split
      · exact h
      · rename_i ne
        apply List.pairwise_cons.mpr
        refine ⟨?_, ih hs⟩
        intro x hx
        rcases (mem_insert n x xs).mp hx with rfl | hx
        · omega
        · exact ha x hx

def normalize : List Nat → List Nat
  | [] => []
  | n :: xs => insert n (normalize xs)

@[simp] theorem mem_normalize (x : Nat) (xs : List Nat) :
    x ∈ normalize xs ↔ x ∈ xs := by
  induction xs with
  | nil => simp [normalize]
  | cons n xs ih => simp [normalize, mem_insert, ih]

theorem length_normalize (xs : List Nat) : (normalize xs).length ≤ xs.length := by
  induction xs with
  | nil => simp [normalize]
  | cons n xs ih =>
    have := length_insert n (normalize xs)
    simp only [normalize, List.length_cons]
    omega

theorem sorted_normalize (xs : List Nat) : (normalize xs).Pairwise (· < ·) := by
  induction xs with
  | nil => simp [normalize]
  | cons n xs ih => exact sorted_insert n _ ih

/-- On an increasing list, indices preserve and reflect height comparison. -/
theorem get_le_iff (xs : List Nat) (hs : xs.Pairwise (· < ·))
    (i j : Fin xs.length) : xs[i.val] ≤ xs[j.val] ↔ i.val ≤ j.val := by
  have h := List.pairwise_iff_getElem.mp hs
  constructor
  · intro hij
    by_cases hn : i.val ≤ j.val
    · exact hn
    · have := h j.val i.val j.isLt i.isLt (by omega)
      omega
  · intro hij
    by_cases heq : i.val = j.val
    · simp [heq]
    · have := h i.val j.val i.isLt j.isLt (by omega)
      omega

theorem idxOf_get (xs : List Nat) (hs : xs.Pairwise (· < ·)) (i : Fin xs.length) :
    xs.idxOf xs[i.val] = i.val := by
  have hm : xs[i.val] ∈ xs := List.getElem_mem i.isLt
  have hr := List.idxOf_lt_length_of_mem hm
  have he := List.getElem_idxOf hr
  have hi := get_le_iff xs hs ⟨xs.idxOf xs[i.val], hr⟩ i
  have hj := get_le_iff xs hs i ⟨xs.idxOf xs[i.val], hr⟩
  simp only [he] at hi hj
  change (_ ↔ xs.idxOf xs[i.val] ≤ i.val) at hi
  change (_ ↔ i.val ≤ xs.idxOf xs[i.val]) at hj
  omega

theorem rank_le_iff (xs : List Nat) (hs : xs.Pairwise (· < ·))
    {a b : Nat} (ha : a ∈ xs) (hb : b ∈ xs) :
    xs.idxOf a ≤ xs.idxOf b ↔ a ≤ b := by
  have h := get_le_iff xs hs
    ⟨xs.idxOf a, List.idxOf_lt_length_of_mem ha⟩
    ⟨xs.idxOf b, List.idxOf_lt_length_of_mem hb⟩
  simpa only [List.getElem_idxOf (List.idxOf_lt_length_of_mem ha),
    List.getElem_idxOf (List.idxOf_lt_length_of_mem hb)] using h.symm

theorem rank_eq_iff (xs : List Nat) (hs : xs.Pairwise (· < ·))
    {a b : Nat} (ha : a ∈ xs) (hb : b ∈ xs) :
    xs.idxOf a = xs.idxOf b ↔ a = b := by
  constructor
  · intro h
    have := (rank_le_iff xs hs ha hb).mp (Nat.le_of_eq h)
    have := (rank_le_iff xs hs hb ha).mp (Nat.le_of_eq h.symm)
    omega
  · intro h; rw [h]

theorem rank_zero (xs : List Nat) (hs : xs.Pairwise (· < ·)) (hz : 0 ∈ xs) :
    xs.idxOf 0 = 0 := by
  have hn : 0 < xs.length := Nat.lt_of_le_of_lt (Nat.zero_le _) (List.idxOf_lt_length_of_mem hz)
  have hm : xs[0] ∈ xs := List.getElem_mem hn
  have h := (rank_le_iff xs hs hz hm).mpr (Nat.zero_le _)
  rw [idxOf_get xs hs ⟨0, hn⟩] at h
  change xs.idxOf 0 ≤ 0 at h
  omega

end RoboVerify.FiniteLevels
