import RoboVerify.FiniteDecision

set_option autoImplicit false

namespace RoboVerify.GroundEPR
open Fragments FiniteDecision
variable {V D A : Type}

def allOf (xs : List A) (f : A → QF V) : QF V :=
  xs.foldr (fun x q => .conj (f x) q) .truth

def anyOf (xs : List A) (f : A → QF V) : QF V :=
  xs.foldr (fun x q => .disj (f x) q) .falsity

theorem eval_allOf (xs : List A) (f : A → QF V) (M : Relation → D → D → Prop) (ρ : V → D) :
    (allOf xs f).eval M ρ ↔ ∀ x ∈ xs, (f x).eval M ρ := by
  induction xs <;> simp_all [allOf, QF.eval]

theorem eval_anyOf (xs : List A) (f : A → QF V) (M : Relation → D → D → Prop) (ρ : V → D) :
    (anyOf xs f).eval M ρ ↔ ∃ x ∈ xs, (f x).eval M ρ := by
  induction xs <;> simp_all [anyOf, QF.eval]

theorem eval_allEnum (xs : Enumeration A) (f : A → QF V)
    (M : Relation → D → D → Prop) (ρ : V → D) :
    (allOf xs.values f).eval M ρ ↔ ∀ x, (f x).eval M ρ := by
  rw [eval_allOf]
  exact ⟨fun h x => h x (xs.covers x), fun h x _ => h x⟩

theorem eval_anyEnum (xs : Enumeration A) (f : A → QF V)
    (M : Relation → D → D → Prop) (ρ : V → D) :
    (anyOf xs.values f).eval M ρ ↔ ∃ x, (f x).eval M ρ := by
  rw [eval_anyOf]
  exact ⟨fun ⟨x, _, h⟩ => ⟨x, h⟩, fun ⟨x, h⟩ => ⟨x, xs.covers x, h⟩⟩

def relations : Enumeration Relation :=
  ⟨[.on, .higher, .scattered, .onZero, .eq], fun r => by cases r <;> simp⟩

/-- A finite atomic diagram uses exactly the assertion vocabulary. It fixes
all relational facts among the chosen object constants, including equality. -/
def diagram {n : Nat} (s f : Scene (Fin n)) : QF (Object (Fin n)) :=
  let objects := (Enumeration.fin n).option
  allOf relations.values fun r => allOf objects.values fun x => allOf objects.values fun y =>
    if atom s f r x y then .atom r x y else .neg (.atom r x y)

theorem diagram_correct {n : Nat} (s f : Scene (Fin n))
    (M : Relation → D → D → Prop) (ρ : Object (Fin n) → D) :
    (diagram s f).eval M ρ ↔
      ∀ r x y, M r (ρ x) (ρ y) ↔ relation s f r x y := by
  simp only [diagram, eval_allEnum]
  apply forall_congr'; intro r
  apply forall_congr'; intro x
  apply forall_congr'; intro y
  have h := atom_correct s f r x y
  cases ha : atom s f r x y <;> simp_all [QF.eval]

/-- Expand only the finite object quantifiers justified by the small-model
theorem. The matrix remains a formula; it is not decided before the reduction. -/
def ground {n : Nat} {W : Type} (ρ : W → Object (Fin n)) (p : EA W) : QF (Object (Fin n)) :=
  let objects := (Enumeration.fin n).option
  anyOf (objects.block p.e).values fun y => allOf (objects.block p.u).values fun x =>
    p.matrix.map (valuation ρ x y)

theorem ground_correct {n : Nat} {W : Type} (s f : Scene (Fin n))
    (ρ : W → Object (Fin n)) (p : EA W)
    (M : Relation → D → D → Prop) (η : Object (Fin n) → D)
    (hd : (diagram s f).eval M η) :
    (ground ρ p).eval M η ↔ p.eval (relation s f) ρ := by
  simp only [ground, eval_anyEnum, eval_allEnum, QF.eval_map, EA.eval]
  apply exists_congr; intro y
  apply forall_congr'; intro x
  exact (QF.eval_embedding p.matrix M (relation s f) η
    (fun r a b => ((diagram_correct s f M η).mp hd r a b).symm) _).symm

def clause {n v : Nat} (s f : Scene (Fin n)) (store : Fin v → Object (Fin n))
    (p : EA (Option (Fin v))) : QF (Object (Fin n)) :=
  if wellFormed s && wellFormed f then .conj (diagram s f) (ground (names store) p) else .falsity

theorem clause_correct {n v : Nat} (s f : Scene (Fin n)) (store : Fin v → Object (Fin n))
    (p : EA (Option (Fin v))) (M : Relation → D → D → Prop) (η : Object (Fin n) → D) :
    (clause s f store p).eval M η ↔
      s.WellFormed ∧ f.WellFormed ∧ (diagram s f).eval M η ∧
        p.eval (relation s f) (names store) := by
  have hw : (wellFormed s && wellFormed f) = true ↔ s.WellFormed ∧ f.WellFormed := by
    simp only [Bool.and_eq_true, wellFormed_correct]
  by_cases h : (wellFormed s && wellFormed f) = true
  · obtain ⟨hs, hf⟩ := hw.mp h
    simp only [clause, h, ite_true, QF.eval, hs, hf, true_and]
    exact and_congr_right fun hd => ground_correct s f (names store) p M η hd
  · have hh : ¬(s.WellFormed ∧ f.WellFormed) := fun hh => h (hw.mpr hh)
    have hc : clause s f store p = .falsity := by simp [clause, h]
    rw [hc]
    change False ↔ _
    constructor
    · intro hh; exact hh.elim
    · rintro ⟨hs, hf, _⟩; exact hh ⟨hs, hf⟩

def includeObject {limit : Nat} (n : Fin (limit + 1)) : Object (Fin n.val) → Object (Fin limit) :=
  Option.map (Fin.castLE (Nat.le_of_lt_succ n.isLt))

def projectObject {limit : Nat} (n : Fin (limit + 1)) : Object (Fin limit) → Object (Fin n.val)
  | none => none
  | some i => if h : i.val < n.val then some ⟨i.val, h⟩ else none

theorem project_include {limit : Nat} (n : Fin (limit + 1)) (x : Object (Fin n.val)) :
    projectObject n (includeObject n x) = x := by
  cases x with
  | none => rfl
  | some i => simp [includeObject, projectObject, i.isLt]

def query {v : Nat} (limit : Nat) (p : EA (Option (Fin v))) : QF (Object (Fin limit)) :=
  anyOf (Enumeration.fin (limit + 1)).values fun n =>
    anyOf (scenes n.val limit) fun s => anyOf (scenes n.val limit) fun f =>
      anyOf (((Enumeration.fin n.val).option).functions v).values fun store =>
        (clause s f store p).map (includeObject n)

def Satisfiable (q : QF V) : Prop :=
  ∃ (D : Type) (M : Relation → D → D → Prop) (ρ : V → D), q.eval M ρ

theorem query_exact {v : Nat} (limit : Nat) (p : EA (Option (Fin v))) :
    Satisfiable (query limit p) ↔ BlockSmallModel.FiniteCounterexample p limit := by
  constructor
  · rintro ⟨D, M, η, h⟩
    simp only [query, eval_anyOf, QF.eval_map, clause_correct] at h
    obtain ⟨n, _, s, hs, f, hf, store, _, hsw, hfw, _, hp⟩ := h
    exact ⟨n.val, Nat.le_of_lt_succ n.isLt, ⟨s, f, names store⟩,
      ⟨hsw, hfw, rfl⟩, (mem_scenes _ _ s).mp hs, (mem_scenes _ _ f).mp hf, hp⟩
  · rintro ⟨n, hn, t, ht, hs, hf, hp⟩
    let count : Fin (limit + 1) := ⟨n, Nat.lt_succ_of_le hn⟩
    refine ⟨Object (Fin n), relation t.scene t.frozen, projectObject count, ?_⟩
    simp only [query, eval_anyOf, QF.eval_map]
    refine ⟨count, (Enumeration.fin (limit + 1)).covers count,
      t.scene, (mem_scenes n limit _).mpr hs,
      t.frozen, (mem_scenes n limit _).mpr hf, (fun i => t.names (some i)),
      (((Enumeration.fin n).option).functions v).covers _, ?_⟩
    have he : projectObject count ∘ includeObject count = id := funext (project_include count)
    rw [he, clause_correct]
    have hn : names (fun i => t.names (some i)) = t.names := by
      funext x; cases x with
      | none => exact ht.2.2.symm
      | some _ => rfl
    refine ⟨ht.1, ht.2.1, (diagram_correct _ _ _ id).mpr (fun _ _ _ => Iff.rfl), ?_⟩
    rw [hn]; exact hp

def slots : Nat → Block
  | 0 => .empty
  | n + 1 => .pair .one (slots n)

def slotIndex : (n : Nat) → Fin n → (slots n).Index
  | 0 => Fin.elim0
  | n + 1 => Fin.cases (.inl ()) (fun i => .inr (slotIndex n i))

def slotValue : (n : Nat) → (slots n).Index → Fin n
  | 0 => Empty.elim
  | n + 1 => Sum.elim (fun _ => 0) (fun i => (slotValue n i).succ)

theorem slotValue_index (n : Nat) (i : Fin n) : slotValue n (slotIndex n i) = i := by
  induction n with
  | zero => exact Fin.elim0 i
  | succ n ih =>
    refine Fin.cases ?_ (fun j => ?_) i
    · rfl
    · exact congrArg Fin.succ (ih j)

def objectSlots (n : Nat) : Block := .pair .one (slots n)

def objectIndex {n : Nat} : Object (Fin n) → (objectSlots n).Index
  | none => .inl ()
  | some i => .inr (slotIndex n i)

def objectValue {n : Nat} : (objectSlots n).Index → Object (Fin n) :=
  Sum.elim (fun _ => none) (fun i => some (slotValue n i))

theorem objectValue_index {n : Nat} (x : Object (Fin n)) : objectValue (objectIndex x) = x := by
  cases x with
  | none => rfl
  | some i => exact congrArg some (slotValue_index n i)

def qfEA (q : QF V) : EA V := ⟨.empty, .empty, q.map Sum.inl⟩

theorem eval_qfEA (q : QF V) (M : Relation → D → D → Prop) (ρ : V → D) :
    (qfEA q).eval M ρ ↔ q.eval M ρ := by
  simp only [qfEA, EA.eval, QF.eval_map]
  constructor
  · rintro ⟨_, h⟩; exact h Empty.elim
  · intro h; exact ⟨Empty.elim, fun _ => h⟩

/-- A closed, function-free existential sentence, hence an EPR query. Its
matrix is the explicitly grounded scene/VC formula, not the answer of a solver. -/
def closed {v : Nat} (limit : Nat) (p : EA (Option (Fin v))) : EA Empty :=
  EA.close (objectSlots limit) (qfEA ((query limit p).map objectIndex))

theorem closed_eval {v : Nat} (limit : Nat) (p : EA (Option (Fin v)))
    (M : Relation → D → D → Prop) :
    (closed limit p).eval M Empty.elim ↔ ∃ η, (query limit p).eval M η := by
  simp only [closed, EA.eval_close, eval_qfEA, QF.eval_map]
  constructor
  · rintro ⟨ρ, h⟩; exact ⟨ρ ∘ objectIndex, h⟩
  · rintro ⟨η, h⟩
    refine ⟨η ∘ objectValue, ?_⟩
    have he : (η ∘ objectValue) ∘ objectIndex = η := by
      funext x; exact congrArg η (objectValue_index x)
    rw [he]; exact h

def EPRSatisfiable (p : EA Empty) : Prop :=
  ∃ (D : Type) (M : Relation → D → D → Prop), p.eval M Empty.elim

theorem closed_exact {v : Nat} (limit : Nat) (p : EA (Option (Fin v))) :
    EPRSatisfiable (closed limit p) ↔ BlockSmallModel.FiniteCounterexample p limit := by
  simp only [EPRSatisfiable, closed_eval]
  exact query_exact limit p

def reduceCounterexample {v : Nat} (p : EA (Option (Fin v))) : EA Empty :=
  closed (BlockSmallModel.bound (v + p.e.size)) p

theorem reduction_exact {v : Nat} (p : EA (Option (Fin v))) :
    EPRSatisfiable (reduceCounterexample p) ↔
      ∃ (α : Type) (s : BlockProgram.State α (Fin v)), BlockProgram.Admissible s ∧
        p.eval (relation s.scene s.frozen) s.names := by
  rw [reduceCounterexample, closed_exact]
  have h := BlockSmallModel.finite_counterexample_iff p (List.finRange v) List.mem_finRange
  simpa only [List.length_finRange] using h.symm

def reduceVC {v : Nat} (vc : AE (Option (Fin v))) : EA Empty := reduceCounterexample vc.refute

/-- Exact validity of a VC over the block model reduces to UNSAT of a closed
EPR sentence. Both SAT and UNSAT have their intended model-relative meaning. -/
theorem vc_reduction_exact {v : Nat} (vc : AE (Option (Fin v))) :
    ¬EPRSatisfiable (reduceVC vc) ↔
      ∀ (α : Type) (s : BlockProgram.State α (Fin v)), BlockProgram.Admissible s →
        vc.eval (relation s.scene s.frozen) s.names := by
  classical
  simp only [reduceVC, reduction_exact, AE.eval_refute, not_exists, not_and, Classical.not_not]

/-- The concrete generator has no assumed transformer or model-encoding
obligation in this theorem. AFR specs/invariants and QF guards are enforced by
the command and specification types; concrete Put preserves AER. -/
theorem generated_vc_reduction {v : Nat} (p q : AF (Option (Fin v)))
    (c : BlockProgram.Command (Fin v)) :
    ∀ vc ∈ BlockProgram.vcgen p q c, ∃ epr : EA Empty,
      (¬EPRSatisfiable epr ↔
        ∀ (α : Type) (s : BlockProgram.State α (Fin v)), BlockProgram.Admissible s →
          vc.eval (relation s.scene s.frozen) s.names) := by
  intro vc _
  exact ⟨reduceVC vc, vc_reduction_exact vc⟩

end RoboVerify.GroundEPR
