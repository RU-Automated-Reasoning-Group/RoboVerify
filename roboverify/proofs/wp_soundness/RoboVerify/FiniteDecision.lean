import RoboVerify.BlockSmallModel

set_option autoImplicit false

namespace RoboVerify.FiniteDecision
open Fragments
variable {D : Type}

/-- Explicit finite enumeration, including the empty type. No classical
proposition-decider is used by the search procedure. -/
structure Enumeration (D : Type) where
  values : List D
  covers : ∀ x, x ∈ values

def Enumeration.fin (n : Nat) : Enumeration (Fin n) :=
  ⟨List.finRange n, List.mem_finRange⟩

def Enumeration.option (e : Enumeration D) : Enumeration (Option D) where
  values := none :: e.values.map some
  covers x := by cases x with
    | none => exact List.mem_cons_self
    | some x => exact List.mem_cons.mpr (Or.inr (List.mem_map.mpr ⟨x, e.covers x, rfl⟩))

def Enumeration.functions (e : Enumeration D) : (n : Nat) → Enumeration (Fin n → D)
  | 0 => ⟨[Fin.elim0], fun f => by
      have h : f = Fin.elim0 := funext fun i => Fin.elim0 i
      rw [h]; exact List.mem_cons_self⟩
  | n + 1 => {
      values := e.values.flatMap fun d => (e.functions n).values.map (Fin.cases d)
      covers f := by
        apply List.mem_flatMap.mpr
        refine ⟨f 0, e.covers _, List.mem_map.mpr ?_⟩
        refine ⟨fun i => f i.succ, (e.functions n).covers _, ?_⟩
        funext i
        exact Fin.cases rfl (fun _ => rfl) i
    }

def Enumeration.block (e : Enumeration D) : (b : Block) → Enumeration (b.Index → D)
  | .empty => ⟨[Empty.elim], fun f => by
      have h : f = Empty.elim := funext fun i => Empty.elim i
      rw [h]; exact List.mem_cons_self⟩
  | .one => {
      values := e.values.map (fun d (_ : Block.one.Index) => d)
      covers f := by
        apply List.mem_map.mpr
        refine ⟨f (), e.covers _, ?_⟩
        funext i; cases i; rfl
    }
  | .pair a b => {
      values := (e.block a).values.flatMap fun f => (e.block b).values.map (Sum.elim f)
      covers h := by
        apply List.mem_flatMap.mpr
        refine ⟨h ∘ Sum.inl, (e.block a).covers _, List.mem_map.mpr ?_⟩
        refine ⟨h ∘ Sum.inr, (e.block b).covers _, ?_⟩
        funext x; cases x <;> rfl
    }

theorem Enumeration.any_iff (e : Enumeration D) (p : D → Bool) :
    e.values.any p = true ↔ ∃ x, p x = true := by
  simp only [List.any_eq_true]
  exact ⟨fun ⟨x, _, hx⟩ => ⟨x, hx⟩, fun ⟨x, hx⟩ => ⟨x, e.covers x, hx⟩⟩

theorem Enumeration.all_iff (e : Enumeration D) (p : D → Bool) :
    e.values.all p = true ↔ ∀ x, p x = true := by
  simp only [List.all_eq_true]
  exact ⟨fun h x => h x (e.covers x), fun h x _ => h x⟩

def atom {n : Nat} (s f : Scene (Fin n)) (r : Relation) (x y : Object (Fin n)) : Bool :=
  match r, x, y with
  | .eq, x, y => decide (x = y)
  | .on, some x, some y => decide (s.root x = s.root y ∧ s.level y ≤ s.level x)
  | .onZero, some x, some y => decide (f.root x = f.root y ∧ f.level y ≤ f.level x)
  | .higher, some x, some y => decide (s.level y ≤ s.level x)
  | .scattered, some x, some y => decide (s.root x ≠ s.root y)
  | .on, none, none | .onZero, none, none | .higher, none, none => true
  | _, _, _ => false

theorem atom_correct {n : Nat} (s f : Scene (Fin n)) (r : Relation) (x y : Object (Fin n)) :
    atom s f r x y = true ↔ relation s f r x y := by
  cases r <;> cases x <;> cases y <;> simp [atom, relation, Scene.On, Scene.Higher, Scene.Scattered]

def qf {V : Type} {n : Nat} (s f : Scene (Fin n)) (ρ : V → Object (Fin n)) : QF V → Bool
  | .atom r x y => atom s f r (ρ x) (ρ y)
  | .truth => true
  | .falsity => false
  | .neg p => !(qf s f ρ p)
  | .conj p q => qf s f ρ p && qf s f ρ q
  | .disj p q => qf s f ρ p || qf s f ρ q

theorem qf_correct {V : Type} {n : Nat} (s f : Scene (Fin n)) (ρ : V → Object (Fin n))
    (p : QF V) : qf s f ρ p = true ↔ p.eval (relation s f) ρ := by
  induction p with
  | neg p ih =>
    change (!(qf s f ρ p)) = true ↔ ¬p.eval (relation s f) ρ
    rw [← ih]
    cases qf s f ρ p <;> decide
  | _ => simp_all [qf, QF.eval, atom_correct]

def ea {V : Type} {n : Nat} (s f : Scene (Fin n)) (ρ : V → Object (Fin n)) (p : EA V) : Bool :=
  let objects := (Enumeration.fin n).option
  (objects.block p.e).values.any fun y => (objects.block p.u).values.all fun x =>
    qf s f (valuation ρ x y) p.matrix

theorem ea_correct {V : Type} {n : Nat} (s f : Scene (Fin n)) (ρ : V → Object (Fin n))
    (p : EA V) : ea s f ρ p = true ↔ p.eval (relation s f) ρ := by
  simp only [ea, Enumeration.any_iff, Enumeration.all_iff, qf_correct, EA.eval]

def wellFormed {n : Nat} (s : Scene (Fin n)) : Bool :=
  let xs := (Enumeration.fin n).values
  (xs.all fun x => decide (s.root (s.root x) = s.root x)) &&
  (xs.all fun x => decide (s.level (s.root x) = 0)) &&
  (xs.all fun x => xs.all fun y =>
    decide (s.root x = s.root y → s.level x = s.level y → x = y)) &&
  (xs.all fun x => (List.range (s.level x + 1)).all fun k =>
    xs.any fun y => decide (s.root y = s.root x ∧ s.level y = k))

theorem wellFormed_correct {n : Nat} (s : Scene (Fin n)) :
    wellFormed s = true ↔ s.WellFormed := by
  simp only [wellFormed, Bool.and_eq_true, List.all_eq_true, List.any_eq_true,
    List.mem_range, Nat.lt_succ_iff, decide_eq_true_eq, Enumeration.fin, List.mem_finRange,
    true_implies, true_and]
  constructor
  · rintro ⟨⟨⟨hr, hz⟩, hu⟩, hc⟩; exact ⟨hr, hz, hu, hc⟩
  · intro h; exact ⟨⟨⟨h.root_fixed, h.root_zero⟩, h.unique⟩, h.complete⟩

def scenes (n limit : Nat) : List (Scene (Fin n)) :=
  ((Enumeration.fin n).functions n).values.flatMap fun root =>
    ((Enumeration.fin (limit + 1)).functions n).values.map fun level =>
      ⟨root, fun i => (level i).val⟩

theorem mem_scenes (n limit : Nat) (s : Scene (Fin n)) :
    s ∈ scenes n limit ↔ ∀ i, s.level i ≤ limit := by
  simp only [scenes, List.mem_flatMap, List.mem_map]
  constructor
  · rintro ⟨root, _, level, _, rfl⟩ i
    change (level i).val ≤ limit
    have := (level i).isLt; omega
  · intro h
    let level (i : Fin n) : Fin (limit + 1) := ⟨s.level i, by have := h i; omega⟩
    refine ⟨s.root, ((Enumeration.fin n).functions n).covers _, level,
      ((Enumeration.fin (limit + 1)).functions n).covers _, ?_⟩
    cases s; rfl

def names {v n : Nat} (store : Fin v → Object (Fin n)) : Option (Fin v) → Object (Fin n)
  | none => none
  | some i => store i

def atSize {v : Nat} (n limit : Nat) (p : EA (Option (Fin v))) : Bool :=
  (scenes n limit).any fun s => (scenes n limit).any fun f =>
    (((Enumeration.fin n).option).functions v).values.any fun store =>
      wellFormed s && wellFormed f && ea s f (names store) p

theorem atSize_correct {v : Nat} (n limit : Nat) (p : EA (Option (Fin v))) :
    atSize n limit p = true ↔
      ∃ t : BlockProgram.State (Fin n) (Fin v), BlockProgram.Admissible t ∧
        (∀ i, t.scene.level i ≤ limit) ∧ (∀ i, t.frozen.level i ≤ limit) ∧
        p.eval (relation t.scene t.frozen) t.names := by
  simp only [atSize, List.any_eq_true, Bool.and_eq_true,
    wellFormed_correct, ea_correct, mem_scenes]
  constructor
  · rintro ⟨s, hs, f, hf, store, _, ⟨hsw, hfw⟩, hp⟩
    exact ⟨⟨s, f, names store⟩, ⟨hsw, hfw, rfl⟩, hs, hf, hp⟩
  · rintro ⟨t, ht, hs, hf, hp⟩
    have hn : names (fun i => t.names (some i)) = t.names := by
      funext x; cases x with
      | none => exact ht.2.2.symm
      | some _ => rfl
    refine ⟨t.scene, hs, t.frozen, hf, (fun i => t.names (some i)), ?_, ⟨ht.1, ht.2.1⟩, ?_⟩
    · exact (((Enumeration.fin n).option).functions v).covers _
    · rw [hn]; exact hp

def search {v : Nat} (limit : Nat) (p : EA (Option (Fin v))) : Bool :=
  (List.range (limit + 1)).any fun n => atSize n limit p

theorem search_correct {v : Nat} (limit : Nat) (p : EA (Option (Fin v))) :
    search limit p = true ↔ BlockSmallModel.FiniteCounterexample p limit := by
  simp only [search, List.any_eq_true, List.mem_range, Nat.lt_succ_iff, atSize_correct,
    BlockSmallModel.FiniteCounterexample]

def counterexample {v : Nat} (p : EA (Option (Fin v))) : Bool :=
  search (BlockSmallModel.bound (v + p.e.size)) p

theorem counterexample_exact {v : Nat} (p : EA (Option (Fin v))) :
    counterexample p = true ↔
      ∃ (α : Type) (s : BlockProgram.State α (Fin v)), BlockProgram.Admissible s ∧
        p.eval (relation s.scene s.frozen) s.names := by
  rw [counterexample, search_correct]
  have h := BlockSmallModel.finite_counterexample_iff p (List.finRange v) List.mem_finRange
  simpa only [List.length_finRange] using h.symm

def valid {v : Nat} (p : AE (Option (Fin v))) : Bool := !counterexample p.refute

theorem valid_exact {v : Nat} (p : AE (Option (Fin v))) :
    valid p = true ↔ ∀ (α : Type) (s : BlockProgram.State α (Fin v)),
      BlockProgram.Admissible s → p.eval (relation s.scene s.frozen) s.names := by
  classical
  have hb : valid p = true ↔ counterexample p.refute ≠ true := by
    unfold valid
    cases counterexample p.refute <;> decide
  rw [hb]
  simp only [ne_eq, counterexample_exact,
    AE.eval_refute, not_exists, not_and, Classical.not_not]

def check {v : Nat} (p q : AF (Option (Fin v))) (c : BlockProgram.Command (Fin v)) : Bool :=
  (BlockProgram.vcgen p q c).all valid

theorem check_exact {v : Nat} (p q : AF (Option (Fin v))) (c : BlockProgram.Command (Fin v)) :
    check p q c = true ↔ ∀ vc ∈ BlockProgram.vcgen p q c,
      ∀ (α : Type) (s : BlockProgram.State α (Fin v)), BlockProgram.Admissible s →
        vc.eval (relation s.scene s.frozen) s.names := by
  simp only [check, List.all_eq_true, valid_exact]

/-- The executable checker decides exactly validity of the generated VCs.
Success entails partial correctness and no modeled fault. This is not a
decision procedure for termination or for the existence of suitable invariants. -/
theorem verified_program {v : Nat} (p q : AF (Option (Fin v)))
    (c : BlockProgram.Command (Fin v)) (checked : check p q c = true)
    {α : Type} [DecidableEq α] (s : BlockProgram.State α (Fin v))
    (hs : BlockProgram.Admissible s) (hp : p.eval (relation s.scene s.frozen) s.names) :
    (¬Fragments.Program.Fault BlockProgram.Operation.Step c.compile s) ∧
      ∀ t, Fragments.Program.Exec BlockProgram.Operation.Step c.compile s t →
        BlockProgram.Admissible t ∧ q.eval (relation t.scene t.frozen) t.names :=
  BlockProgram.verified_program p q c
    (fun vc hv t ht => (check_exact p q c).mp checked vc hv α t ht) s hs hp

end RoboVerify.FiniteDecision
