import RoboVerify.SceneCompression
import RoboVerify.FiniteRestriction
import RoboVerify.FiniteModel
import RoboVerify.BlockProgram

set_option autoImplicit false

namespace RoboVerify.BlockSmallModel
open Fragments
variable {α V : Type}

def bound (k : Nat) : Nat := k * (k + 1)

theorem le_bound (k : Nat) : k ≤ bound k := by
  have h := Nat.mul_le_mul_left k (Nat.le_add_left 1 k)
  simpa [bound] using h

theorem bound_mono {k l : Nat} (h : k ≤ l) : bound k ≤ bound l :=
  Nat.mul_le_mul h (Nat.add_le_add_right h 1)

/-- A genuine finite pair of well-formed scenes, embedded in the original
relational structure. Every retained atomic fact, including equality, is exact. -/
structure Submodel (s f : Scene α) (seeds : List α) where
  size : Nat
  size_bound : size ≤ bound seeds.length
  current : Scene (Fin size)
  frozen : Scene (Fin size)
  current_wf : current.WellFormed
  frozen_wf : frozen.WellFormed
  embed : Fin size → α
  injective : ∀ x y, embed x = embed y → x = y
  included : ∀ a ∈ seeds, ∃ i, embed i = a
  exact_atoms : ∀ r x y, relation current frozen r x y ↔
    relation s f r (x.map embed) (y.map embed)
  current_bound : ∀ i, current.level i ≤ bound seeds.length
  frozen_bound : ∀ i, frozen.level i ≤ bound seeds.length

noncomputable def submodel (s f : Scene α) (hs : s.WellFormed) (hf : f.WellFormed)
    (seeds : List α) : Submodel s f seeds := by
  classical
  let β := SceneCompression.Domain s hs seeds
  let selected := SceneCompression.retained s hs seeds (SceneCompression.levels s seeds)
  let objects : List β := selected.attach
  have covers : ∀ x : β, x ∈ objects := List.mem_attach selected
  let embedding : β → α := Subtype.val
  have inj : ∀ x y : β, embedding x = embedding y → x = y := fun _ _ => Subtype.ext
  let current := SceneCompression.compressed s hs seeds
  let frozen := FiniteRestriction.scene f embedding objects covers
  let code := FiniteModel.Code.ofList objects covers
  have hcur := SceneCompression.compressed_wellFormed s hs seeds
  have hfroz := FiniteRestriction.wellFormed f hf embedding inj objects covers
  have hlen : objects.length ≤ bound seeds.length := by
    have he : objects.length = selected.length := List.length_attach
    rw [he]
    exact SceneCompression.retained_bound s hs seeds
  have atoms (r : Relation) (x y : Object β) :
      relation current frozen r x y ↔ relation s f r (x.map embedding) (y.map embedding) := by
    cases r <;> cases x <;> cases y <;>
      simp only [relation, Option.map_none, Option.map_some, reduceCtorEq]
    · exact SceneCompression.compressed_on s hs seeds _ _
    · exact SceneCompression.compressed_higher s hs seeds _ _
    · exact SceneCompression.compressed_scattered s hs seeds _ _
    · exact FiniteRestriction.on_exact f hf embedding inj objects covers _ _
    · simp only [Option.some.injEq]
      exact ⟨congrArg Subtype.val, Subtype.ext⟩
  refine {
    size := code.size
    size_bound := Nat.le_trans (FiniteModel.Code.ofList_bound objects covers) hlen
    current := code.scene current
    frozen := code.scene frozen
    current_wf := code.scene_wellFormed current hcur
    frozen_wf := code.scene_wellFormed frozen hfroz
    embed := embedding ∘ code.decode
    injective := fun x y h => code.decode_inj.mp (inj _ _ h)
    included := ?_
    exact_atoms := ?_
    current_bound := ?_
    frozen_bound := ?_
  }
  · intro a ha
    let x : β := ⟨a, SceneCompression.seed_mem s hs seeds ha⟩
    exact ⟨code.encode x, by simp only [Function.comp_apply, code.decode_encode]; rfl⟩
  · intro r x y
    have h := (code.relation_decode current frozen r x y).trans
      (atoms r (x.map code.decode) (y.map code.decode))
    simpa only [Option.map_map] using h
  · intro i
    exact Nat.le_trans (SceneCompression.compressed_level_bound s hs seeds (code.decode i))
      (le_bound seeds.length)
  · intro i
    exact Nat.le_trans (Nat.le_of_lt (FiniteRestriction.level_bound f embedding objects covers
      (code.decode i))) hlen

namespace Submodel

noncomputable def lift {s f : Scene α} {seeds : List α} (m : Submodel s f seeds) :
    Object α → Object (Fin m.size) := by
  classical
  exact fun x => match x with
    | none => none
    | some a => if h : ∃ i, m.embed i = a then some h.choose else none

@[simp] theorem lift_none {s f : Scene α} {seeds : List α} (m : Submodel s f seeds) :
    m.lift none = none := rfl

theorem lift_included {s f : Scene α} {seeds : List α} (m : Submodel s f seeds)
    (a : α) (ha : a ∈ seeds) : (m.lift (some a)).map m.embed = some a := by
  classical
  have h := m.included a ha
  simp only [lift, dite_eq_left h, Option.map_some, h.choose_spec]

end Submodel

/-- The finite bound counts declared program names and existential variables,
not the number of blocks or heights in the original scene. Universal variables
do not need to be retained: the universal matrix survives an induced submodel. -/
theorem small_counterexample (p : EA (Option V)) (variables : List V)
    (covers : ∀ v, v ∈ variables) (s : BlockProgram.State α V)
    (hs : BlockProgram.Admissible s) (h : p.eval (relation s.scene s.frozen) s.names) :
    ∃ n, n ≤ bound (variables.length + p.e.size) ∧
      ∃ t : BlockProgram.State (Fin n) V, BlockProgram.Admissible t ∧
        (∀ i, t.scene.level i ≤ bound (variables.length + p.e.size)) ∧
        (∀ i, t.frozen.level i ≤ bound (variables.length + p.e.size)) ∧
        p.eval (relation t.scene t.frozen) t.names := by
  classical
  obtain ⟨y, hy⟩ := h
  let picked := variables.map (fun v => s.names (some v)) ++ p.e.enumerate.map y
  let seeds := picked.filterMap id
  let m := submodel s.scene s.frozen hs.1 hs.2.1 seeds
  have len : seeds.length ≤ variables.length + p.e.size := by
    have hl := List.length_filterMap_le id picked
    simpa only [picked, List.length_append, List.length_map, Block.enumerate_length] using hl
  have hb := bound_mono len
  have agree (x : Object α) (hx : x ∈ picked) : (m.lift x).map m.embed = x := by
    cases x with
    | none => rfl
    | some a => exact m.lift_included a (List.mem_filterMap.mpr ⟨some a, hx, rfl⟩)
  have names : (fun x => x.map m.embed) ∘ (m.lift ∘ s.names) = s.names := by
    funext v
    cases v with
    | none => change (m.lift (s.names none)).map m.embed = s.names none
              rw [hs.2.2]; rfl
    | some v => exact agree _ (List.mem_append.mpr (Or.inl
        (List.mem_map.mpr ⟨v, covers v, rfl⟩)))
  have witnesses : (fun x => x.map m.embed) ∘ (m.lift ∘ y) = y := by
    funext v
    exact agree _ (List.mem_append.mpr (Or.inr
      (List.mem_map.mpr ⟨v, p.e.enumerate_covers v, rfl⟩)))
  let t : BlockProgram.State (Fin m.size) V :=
    ⟨m.current, m.frozen, m.lift ∘ s.names⟩
  refine ⟨m.size, Nat.le_trans m.size_bound hb, t,
    ⟨m.current_wf, m.frozen_wf, ?_⟩,
    (fun i => Nat.le_trans (m.current_bound i) hb),
    (fun i => Nat.le_trans (m.frozen_bound i) hb), ?_⟩
  · change m.lift (s.names none) = none
    rw [hs.2.2]; rfl
  · exact p.restrict _ _ _ m.exact_atoms s.names (m.lift ∘ s.names) names y (m.lift ∘ y)
      witnesses hy

def FiniteCounterexample (p : EA (Option V)) (limit : Nat) : Prop :=
  ∃ n, n ≤ limit ∧ ∃ t : BlockProgram.State (Fin n) V, BlockProgram.Admissible t ∧
    (∀ i, t.scene.level i ≤ limit) ∧ (∀ i, t.frozen.level i ≤ limit) ∧
    p.eval (relation t.scene t.frozen) t.names

/-- Exactness is in both directions, over the block model, rather than only
soundness of UNSAT in an unconstrained relational overapproximation. -/
theorem finite_counterexample_iff (p : EA (Option V)) (variables : List V)
    (covers : ∀ v, v ∈ variables) :
    (∃ (α : Type) (s : BlockProgram.State α V), BlockProgram.Admissible s ∧
      p.eval (relation s.scene s.frozen) s.names) ↔
    FiniteCounterexample p (bound (variables.length + p.e.size)) := by
  constructor
  · rintro ⟨α, s, hs, h⟩
    exact small_counterexample p variables covers s hs h
  · rintro ⟨n, _, t, ht, _, _, h⟩
    exact ⟨Fin n, t, ht, h⟩

end RoboVerify.BlockSmallModel
