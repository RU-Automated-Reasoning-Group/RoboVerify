import RoboVerify.Fragments

set_option autoImplicit false

namespace RoboVerify

namespace FiniteModel
variable {α : Type}

def unique [DecidableEq α] : List α → List α
  | [] => []
  | x :: xs => let ys := unique xs; if x ∈ ys then ys else x :: ys

@[simp] theorem mem_unique [DecidableEq α] (x : α) (xs : List α) :
    x ∈ unique xs ↔ x ∈ xs := by
  induction xs generalizing x with
  | nil => simp [unique]
  | cons a xs ih =>
    simp only [unique]
    split
    · rename_i h
      have ha := (ih a).mp h
      simp_all
    · simp [ih]

theorem unique_nodup [DecidableEq α] (xs : List α) : (unique xs).Nodup := by
  induction xs with
  | nil => simp [unique]
  | cons a xs ih =>
    simp only [unique]
    split
    · exact ih
    · rename_i h; exact List.nodup_cons.mpr ⟨h, ih⟩

theorem unique_length [DecidableEq α] (xs : List α) : (unique xs).length ≤ xs.length := by
  induction xs with
  | nil => simp [unique]
  | cons a xs ih =>
    simp only [unique]
    split <;> simp only [List.length_cons] <;> omega

structure Code (α : Type) where
  size : Nat
  encode : α → Fin size
  decode : Fin size → α
  decode_encode : ∀ x, decode (encode x) = x
  encode_decode : ∀ i, encode (decode i) = i

namespace Code

theorem encode_inj (c : Code α) {x y : α} : c.encode x = c.encode y ↔ x = y := by
  constructor
  · intro h; simpa only [c.decode_encode] using congrArg c.decode h
  · intro h; rw [h]

theorem decode_inj (c : Code α) {i j : Fin c.size} : c.decode i = c.decode j ↔ i = j := by
  constructor
  · intro h; simpa only [c.encode_decode] using congrArg c.encode h
  · intro h; rw [h]

noncomputable def ofList (xs : List α) (covers : ∀ x, x ∈ xs) : Code α := by
  classical
  let ys := unique xs
  have hm (x : α) : x ∈ ys := (mem_unique x xs).mpr (covers x)
  exact {
    size := ys.length
    encode x := ⟨ys.idxOf x, List.idxOf_lt_length_of_mem (hm x)⟩
    decode i := ys[i.val]
    decode_encode x := List.getElem_idxOf (List.idxOf_lt_length_of_mem (hm x))
    encode_decode i := Fin.ext ((unique_nodup xs).idxOf_getElem i.val i.isLt)
  }

theorem ofList_bound (xs : List α) (covers : ∀ x, x ∈ xs) :
    (ofList xs covers).size ≤ xs.length := by
  classical
  exact unique_length xs

def scene (c : Code α) (s : Scene α) : Scene (Fin c.size) where
  root i := c.encode (s.root (c.decode i))
  level i := s.level (c.decode i)

theorem scene_wellFormed (c : Code α) (s : Scene α) (wf : s.WellFormed) :
    (c.scene s).WellFormed := by
  constructor
  · intro i; simp only [scene, c.decode_encode, wf.root_fixed]
  · intro i; simp only [scene, c.decode_encode, wf.root_zero]
  · intro i j hr hl
    apply c.decode_inj.mp
    apply wf.unique
    · exact c.encode_inj.mp hr
    · exact hl
  · intro i k hk
    obtain ⟨x, hr, hl⟩ := wf.complete (c.decode i) k hk
    exact ⟨c.encode x, by simp only [scene, c.decode_encode, hr],
      by simpa only [scene, c.decode_encode] using hl⟩

theorem relation_decode (c : Code α) (s f : Scene α) (r : Relation)
    (x y : Object (Fin c.size)) :
    relation (c.scene s) (c.scene f) r x y ↔
      relation s f r (x.map c.decode) (y.map c.decode) := by
  cases r <;> cases x <;> cases y <;>
    simp [relation, Scene.On, Scene.Higher, Scene.Scattered, scene,
      c.encode_inj, c.decode_inj]

end Code
end FiniteModel

namespace Fragments
namespace QF
variable {V D E : Type}

theorem eval_embedding (q : QF V) (M : Relation → D → D → Prop)
    (N : Relation → E → E → Prop) (f : E → D)
    (correct : ∀ r x y, N r x y ↔ M r (f x) (f y)) (ρ : V → E) :
    q.eval N ρ ↔ q.eval M (f ∘ ρ) := by
  induction q <;> simp_all [eval, Function.comp_def]

end QF

theorem EA.restrict {V D E : Type} (p : EA V)
    (M : Relation → D → D → Prop) (N : Relation → E → E → Prop) (f : E → D)
    (correct : ∀ r x y, N r x y ↔ M r (f x) (f y))
    (ρ : V → D) (ρ' : V → E) (names : f ∘ ρ' = ρ)
    (y : p.e.Index → D) (y' : p.e.Index → E) (witnesses : f ∘ y' = y)
    (h : ∀ x, p.matrix.eval M (valuation ρ x y)) : p.eval N ρ' := by
  refine ⟨y', fun x => ?_⟩
  apply (QF.eval_embedding p.matrix M N f correct _).mpr
  have he : f ∘ valuation ρ' x y' = valuation ρ (f ∘ x) y := by
    funext v
    cases v with
    | inl v => exact congrFun names v
    | inr v => cases v with
      | inl v => rfl
      | inr v => exact congrFun witnesses v
  rw [he]; exact h _

def Block.size : Block → Nat
  | .empty => 0
  | .one => 1
  | .pair a b => a.size + b.size

def Block.enumerate : (b : Block) → List b.Index
  | .empty => []
  | .one => [()]
  | .pair a b => (a.enumerate.map Sum.inl) ++ (b.enumerate.map Sum.inr)

theorem Block.enumerate_covers (b : Block) (x : b.Index) : x ∈ b.enumerate := by
  induction b with
  | empty => exact Empty.elim x
  | one => cases x; exact List.mem_cons_self
  | pair a b ha hb =>
    cases x with
    | inl x =>
      change Sum.inl x ∈ (a.enumerate.map Sum.inl) ++ (b.enumerate.map Sum.inr)
      exact List.mem_append.mpr (Or.inl (List.mem_map.mpr ⟨x, ha x, rfl⟩))
    | inr x =>
      change Sum.inr x ∈ (a.enumerate.map Sum.inl) ++ (b.enumerate.map Sum.inr)
      exact List.mem_append.mpr (Or.inr (List.mem_map.mpr ⟨x, hb x, rfl⟩))

theorem Block.enumerate_length (b : Block) : b.enumerate.length = b.size := by
  induction b with
  | empty => rfl
  | one => rfl
  | pair a b ha hb =>
    change ((a.enumerate.map Sum.inl) ++ (b.enumerate.map Sum.inr)).length = a.size + b.size
    simp only [List.length_append, List.length_map, ha, hb]

end Fragments
end RoboVerify
