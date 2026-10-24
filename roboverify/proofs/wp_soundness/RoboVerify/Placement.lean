import RoboVerify.FragmentVC
import RoboVerify.FormulaBridge
import RoboVerify.Program
import RoboVerify.WitnessWP

set_option autoImplicit false

namespace RoboVerify.Fragments

namespace QF
variable {V D : Type}

def imp (p q : QF V) : QF V := .disj (.neg p) q
def ite (p q r : QF V) : QF V := .disj (.conj p q) (.conj (.neg p) r)

@[simp] theorem eval_imp (p q : QF V) (M : Relation → D → D → Prop) (ρ : V → D) :
    (imp p q).eval M ρ ↔ (p.eval M ρ → q.eval M ρ) := by
  classical
  by_cases h : p.eval M ρ <;> simp [imp, eval, h]

end QF

namespace AF
variable {V D : Type}

@[simp] theorem eval_conj (p q : AF V) (M : Relation → D → D → Prop) (ρ : V → D) :
    (AF.conj p q).eval M ρ ↔ p.eval M ρ ∧ q.eval M ρ := Iff.rfl

theorem eval_all_one (p : QF (V ⊕ Block.one.Index))
    (M : Relation → D → D → Prop) (ρ : V → D) :
    (AF.all .one p).eval M ρ ↔ ∀ d, p.eval M (Sum.elim ρ (fun _ => d)) := by
  constructor
  · intro h d; exact h (fun _ => d)
  · intro h x
    have hx : (fun _ : Block.one.Index => x ()) = x := by funext u; cases u; rfl
    simpa only [hx] using h (x ())

end AF

namespace Placement
open QF
variable {V α : Type}

/-- All operands are variables/constants of the relational syntax, including
the distinguished table term. Physical guards handle table-valued aliases. -/
def physical (x tbl : V) : QF V := .neg (.atom .eq x tbl)

def putPhysical (a b : V) (row : V → QF V) (r : Relation) (x y : V) : QF V :=
  match r with
  | .on => .disj (.conj (.neg (.atom .eq x a)) (.atom .on x y))
      (.conj (.atom .eq x a) (.disj (.atom .eq y a) (.atom .on b y)))
  | .higher => ite (.atom .eq x a) (.disj (.atom .eq y a) (row y))
      (ite (.atom .eq y a)
        (.conj (.atom .higher x b) (.neg (.atom .higher b x))) (.atom .higher x y))
  | .scattered => ite (.atom .eq x a)
      (.conj (.neg (.atom .eq y a)) (.atom .scattered b y))
      (ite (.atom .eq y a) (.atom .scattered x b) (.atom .scattered x y))
  | .onZero => .atom .onZero x y
  | .eq => .atom .eq x y

def putAtom (a b tbl : V) (row : V → QF V) (r : Relation) (x y : V) : QF V :=
  ite (.conj (physical x tbl) (physical y tbl)) (putPhysical a b row r x y) (.atom r x y)

def clear (a : V) : AF V := .all .one
  (imp (.atom .on (.inr ()) (.inl a)) (.atom .eq (.inr ()) (.inl a)))

def maximum (b tbl : V) : AF V := .all .one
  (imp (physical (.inr ()) (.inl tbl)) (.atom .higher (.inl b) (.inr ())))

def next (b u : V) : AF V := .conj
  (AF.ofQF (.conj (.atom .higher u b) (.neg (.atom .higher b u))))
  (.all .one (imp
    (.conj (.atom .higher (.inr ()) (.inl b))
      (.neg (.atom .higher (.inl b) (.inr ()))))
    (.atom .higher (.inr ()) (.inl u))))

def legal (a b tbl : V) : AF V :=
  .conj (clear a) (.conj (clear b) (AF.ofQF
    (.conj (physical a tbl) (.conj (physical b tbl) (.neg (.atom .eq a b))))))

def bottom (u tbl : V) : AF V := .conj (AF.ofQF (physical u tbl))
  (.all .one (imp (physical (.inr ()) (.inl tbl))
    (.atom .higher (.inr ()) (.inl u))))

def legalSeparate (a tbl : V) : AF V := .conj (clear a) (AF.ofQF (physical a tbl))

def separatePhysical (a : V) (row : V → QF V) (r : Relation) (x y : V) : QF V :=
  match r with
  | .on => .conj (.atom .on x y)
      (.disj (.neg (.atom .on x a)) (.atom .on y a))
  | .higher => ite (.atom .eq x a) (.disj (.atom .eq y a) (row y))
      (ite (.atom .eq y a) .truth (.atom .higher x y))
  | .scattered => ite (.atom .eq x a) (.neg (.atom .eq y a))
      (ite (.atom .eq y a) .truth (.atom .scattered x y))
  | .onZero => .atom .onZero x y
  | .eq => .atom .eq x y

def separateAtom (a tbl : V) (row : V → QF V) (r : Relation) (x y : V) : QF V :=
  ite (.conj (physical x tbl) (physical y tbl)) (separatePhysical a row r x y) (.atom r x y)

theorem eval_clear (s frozen : Scene α) (ρ : V → Object α) (a : V) (a' : α)
    (ha : ρ a = some a') : (clear a).eval (relation s frozen) ρ ↔ s.Clear a' := by
  rw [clear, AF.eval_all_one]
  simp [QF.eval, relation, ha, Scene.Clear, Option.forall, Sum.elim]

theorem eval_maximum (s frozen : Scene α) (ρ : V → Object α) (b tbl : V) (b' : α)
    (hb : ρ b = some b') (ht : ρ tbl = none) :
    (maximum b tbl).eval (relation s frozen) ρ ↔ HeightWitness.MaxHeight s b' := by
  rw [maximum, AF.eval_all_one]
  simp [physical, QF.eval, relation, hb, ht, HeightWitness.MaxHeight, Option.forall, Sum.elim]

theorem eval_next (s frozen : Scene α) (ρ : V → Object α) (b u : V) (b' u' : α)
    (hb : ρ b = some b') (hu : ρ u = some u') :
    (next b u).eval (relation s frozen) ρ ↔ HeightWitness.NextRep s b' u' := by
  simp only [next, AF.eval_conj, AF.eval_ofQF, AF.eval_all_one]
  simp [QF.eval, relation, hb, hu, HeightWitness.NextRep, Option.forall, Sum.elim, and_assoc]

theorem next_physical (s frozen : Scene α) (ρ : V → Object α) (b u : V) (b' : α)
    (hb : ρ b = some b') (h : (next b u).eval (relation s frozen) ρ) :
    ∃ u', ρ u = some u' := by
  cases hu : ρ u with
  | some u' => exact ⟨u', rfl⟩
  | none =>
    have hh := h.1
    simp [AF.eval_ofQF, QF.eval, relation, hb, hu] at hh

theorem eval_legal (s frozen : Scene α) (ρ : V → Object α) (a b tbl : V) (a' b' : α)
    (ha : ρ a = some a') (hb : ρ b = some b') (ht : ρ tbl = none) :
    (legal a b tbl).eval (relation s frozen) ρ ↔
      s.Clear a' ∧ s.Clear b' ∧ a' ≠ b' := by
  simp only [legal, AF.eval_conj, eval_clear s frozen ρ a a' ha,
    eval_clear s frozen ρ b b' hb, AF.eval_ofQF]
  simp [physical, QF.eval, relation, ha, hb, ht]

theorem eval_bottom (s frozen : Scene α) (ρ : V → Object α) (u tbl : V) (u' : α)
    (hu : ρ u = some u') (ht : ρ tbl = none) :
    (bottom u tbl).eval (relation s frozen) ρ ↔ s.BottomHeight u' := by
  simp only [bottom, AF.eval_conj, AF.eval_ofQF, AF.eval_all_one]
  simp [physical, QF.eval, relation, hu, ht, Scene.BottomHeight, Option.forall, Sum.elim]

theorem bottom_physical (s frozen : Scene α) (ρ : V → Object α) (u tbl : V)
    (ht : ρ tbl = none) (h : (bottom u tbl).eval (relation s frozen) ρ) :
    ∃ u', ρ u = some u' := by
  cases hu : ρ u with
  | some u' => exact ⟨u', rfl⟩
  | none =>
    have hh := h.1
    simp [AF.eval_ofQF, physical, QF.eval, relation, ht, hu] at hh

theorem eval_legalSeparate (s frozen : Scene α) (ρ : V → Object α) (a tbl : V) (a' : α)
    (ha : ρ a = some a') (ht : ρ tbl = none) :
    (legalSeparate a tbl).eval (relation s frozen) ρ ↔ s.Clear a' := by
  simp only [legalSeparate, AF.eval_conj, eval_clear s frozen ρ a a' ha, AF.eval_ofQF]
  simp [physical, QF.eval, relation, ha, ht]

variable [DecidableEq α]

theorem putAtom_correct {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (ρ : V → Object α) (a b tbl : V) (a' b' : α)
    (ha : ρ a = some a') (hb : ρ b = some b') (ht : ρ tbl = none)
    (ca : s.Clear a') (cb : s.Clear b') (row : V → QF V)
    (hr : ∀ v c, ρ v = some c →
      ((row v).eval (relation s frozen) ρ ↔ s.NextHeight b' c))
    (r : Relation) (x y : V) :
    (putAtom a b tbl row r x y).eval (relation s frozen) ρ ↔
      relation (s.put a' b') frozen r (ρ x) (ρ y) := by
  rw [← Formula.putAtom_correct wf frozen ca cb r (ρ x) (ρ y)]
  cases hx : ρ x with
  | none =>
    cases hy : ρ y <;> cases r <;>
      simp [putAtom, physical, putPhysical, QF.ite, QF.eval, relation,
        Formula.putAtom, Formula.eval, ha, hb, ht, hx, hy]
  | some x' =>
    cases hy : ρ y with
    | none =>
      cases r <;>
        simp [putAtom, physical, putPhysical, QF.ite, QF.eval, relation,
          Formula.putAtom, Formula.eval, ha, hb, ht, hx, hy]
    | some y' =>
      have hrow := hr y y' hy
      cases r <;>
        simp [putAtom, physical, putPhysical, QF.ite, QF.eval, relation,
          Formula.putAtom, Formula.putPhysical, Formula.iteF, Formula.phys,
          Formula.eval, ha, hb, ht, hx, hy, hrow]

theorem separateAtom_correct {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (ρ : V → Object α) (a tbl : V) (a' : α)
    (ha : ρ a = some a') (ht : ρ tbl = none) (ca : s.Clear a') (row : V → QF V)
    (hr : ∀ v c, ρ v = some c → ((row v).eval (relation s frozen) ρ ↔ s.BottomHeight c))
    (r : Relation) (x y : V) :
    (separateAtom a tbl row r x y).eval (relation s frozen) ρ ↔
      relation (s.separate a') frozen r (ρ x) (ρ y) := by
  rw [← Formula.separateAtom_correct wf frozen ca r (ρ x) (ρ y)]
  cases hx : ρ x with
  | none =>
    cases hy : ρ y <;> cases r <;>
      simp [separateAtom, physical, separatePhysical, QF.ite, QF.eval, relation,
        Formula.separateAtom, Formula.eval, ha, ht, hx, hy]
  | some x' =>
    cases hy : ρ y with
    | none =>
      cases r <;>
        simp [separateAtom, physical, separatePhysical, QF.ite, QF.eval, relation,
          Formula.separateAtom, Formula.eval, ha, ht, hx, hy]
    | some y' =>
      have hrow := hr y y' hy
      cases r <;>
        simp [separateAtom, physical, separatePhysical, QF.ite, QF.eval, relation,
          Formula.separateAtom, Formula.separatePhysical, Formula.iteF, Formula.phys,
          Formula.eval, ha, ht, hx, hy, hrow]

/-- Both choices replace every atom in the matrix and retain the input
forall-exists prefix. The selector quantifiers are added by wpBlock below. -/
def rewritten (a b tbl : V) (representative : Option V) (q : AE V) : AE V :=
  q.replace (putAtom (.inl a) (.inl b) (.inl tbl) fun y =>
    match representative with
    | none => .truth
    | some u => .atom .higher (.inl u) y)

def wpBlock (a b tbl : V) (q : AE V) : AE V :=
  Rules.sharedPut .one (legal a b tbl) (maximum b tbl)
    (next (.inl b) (.inr ())) (rewritten a b tbl none q)
    (rewritten (.inl a) (.inl b) (.inl tbl) (some (.inr ())) (q.map Sum.inl))

theorem rewritten_max_correct {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (ρ : V → Object α) (a b tbl : V) (a' b' : α)
    (ha : ρ a = some a') (hb : ρ b = some b') (ht : ρ tbl = none)
    (ca : s.Clear a') (cb : s.Clear b') (hm : HeightWitness.MaxHeight s b') (q : AE V) :
    (rewritten a b tbl none q).eval (relation s frozen) ρ ↔
      q.eval (relation (s.put a' b') frozen) ρ := by
  apply AE.eval_replace
  intro xs ys r x y
  apply putAtom_correct wf frozen (valuation ρ xs ys) (.inl a) (.inl b) (.inl tbl)
    a' b' ha hb ht ca cb
  intro v c _
  rw [Scene.nextHeight_iff wf]
  simp only [QF.eval, true_iff]
  have hc := hm c
  unfold Scene.Higher at hc
  omega

theorem rewritten_rep_correct {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (ρ : V → Object α) (a b tbl u : V) (a' b' u' : α)
    (ha : ρ a = some a') (hb : ρ b = some b') (ht : ρ tbl = none)
    (hu : ρ u = some u') (ca : s.Clear a') (cb : s.Clear b')
    (hn : HeightWitness.NextRep s b' u') (q : AE V) :
    (rewritten a b tbl (some u) q).eval (relation s frozen) ρ ↔
      q.eval (relation (s.put a' b') frozen) ρ := by
  apply AE.eval_replace
  intro xs ys r x y
  apply putAtom_correct wf frozen (valuation ρ xs ys) (.inl a) (.inl b) (.inl tbl)
    a' b' ha hb ht ca cb
  intro v c hv
  change relation s frozen .higher (ρ u) (valuation ρ xs ys v) ↔ s.NextHeight b' c
  rw [hu, hv, Scene.nextHeight_iff wf]
  have hlevel := (HeightWitness.nextRep_iff wf b' u').mp hn
  change s.level c ≤ s.level u' ↔ s.level c ≤ s.level b' + 1
  rw [hlevel]

/-- Concrete AER-to-AER block placement, with no assumed action transformer.
Every relation occurrence and every object/table case is handled above. -/
theorem wpBlock_exact {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (ρ : V → Object α) (a b tbl : V) (a' b' : α)
    (ha : ρ a = some a') (hb : ρ b = some b') (ht : ρ tbl = none) (q : AE V) :
    (wpBlock a b tbl q).eval (relation s frozen) ρ ↔
      (s.Clear a' ∧ s.Clear b' ∧ a' ≠ b') ∧
        q.eval (relation (s.put a' b') frozen) ρ := by
  by_cases hl : s.Clear a' ∧ s.Clear b' ∧ a' ≠ b'
  · have coverage : (maximum b tbl).eval (relation s frozen) ρ ∨
        ∃ z, (next (.inl b) (.inr ())).eval (relation s frozen) (Sum.elim ρ z) := by
      rcases HeightWitness.coverage wf b' with hm | ⟨u', hn⟩
      · exact Or.inl ((eval_maximum s frozen ρ b tbl b' hb ht).mpr hm)
      · exact Or.inr ⟨fun _ => some u',
          (eval_next s frozen (Sum.elim ρ (fun _ : Block.one.Index => some u'))
            (.inl b) (.inr ()) b' u' hb rfl).mpr hn⟩
    have maxCorrect (hm : (maximum b tbl).eval (relation s frozen) ρ) :=
      rewritten_max_correct wf frozen ρ a b tbl a' b' ha hb ht hl.1 hl.2.1
        ((eval_maximum s frozen ρ b tbl b' hb ht).mp hm) q
    have nextCorrect (z : Block.one.Index → Object α)
        (hn : (next (.inl b) (.inr ())).eval (relation s frozen) (Sum.elim ρ z)) :
        (rewritten (.inl a) (.inl b) (.inl tbl) (some (.inr ())) (q.map Sum.inl)).eval
          (relation s frozen) (Sum.elim ρ z) ↔ q.eval (relation (s.put a' b') frozen) ρ := by
      obtain ⟨u', hu⟩ := next_physical s frozen (Sum.elim ρ z) (.inl b) (.inr ()) b' hb hn
      have hrep := (eval_next s frozen (Sum.elim ρ z) (.inl b) (.inr ()) b' u' hb hu).mp hn
      simpa only [AE.eval_map, Function.comp_def, Sum.elim_inl] using
        rewritten_rep_correct wf frozen (Sum.elim ρ z) (.inl a) (.inl b) (.inl tbl)
          (.inr ()) a' b' u' ha hb ht hu hl.1 hl.2.1 hrep (q.map Sum.inl)
    exact (Rules.sharedPut_exact .one (legal a b tbl) (maximum b tbl)
      (next (.inl b) (.inr ())) _ _ (relation s frozen) ρ _ coverage maxCorrect nextCorrect).trans
        (and_congr (eval_legal s frozen ρ a b tbl a' b' ha hb ht) Iff.rfl)
  · simp only [wpBlock, Rules.sharedPut, AE.eval_conj, AF.eval_toAE,
      eval_legal s frozen ρ a b tbl a' b' ha hb ht, hl, false_and]

def rewrittenSeparate (a tbl u : V) (q : AE V) : AE V :=
  q.replace (separateAtom (.inl a) (.inl tbl) fun y => .atom .higher (.inl u) y)

def wpTable (a tbl : V) (q : AE V) : AE V :=
  (legalSeparate a tbl).toAE.conj
    (AE.all .one ((bottom (.inr ()) (.inl tbl)).implies
      (rewrittenSeparate (.inl a) (.inl tbl) (.inr ()) (q.map Sum.inl))))

theorem rewritten_bottom_correct {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (ρ : V → Object α) (a tbl u : V) (a' u' : α)
    (ha : ρ a = some a') (ht : ρ tbl = none) (hu : ρ u = some u')
    (ca : s.Clear a') (hn : s.BottomHeight u') (q : AE V) :
    (rewrittenSeparate a tbl u q).eval (relation s frozen) ρ ↔
      q.eval (relation (s.separate a') frozen) ρ := by
  apply AE.eval_replace
  intro xs ys r x y
  apply separateAtom_correct wf frozen (valuation ρ xs ys) (.inl a) (.inl tbl) a' ha ht ca
  intro v c hv
  change relation s frozen .higher (ρ u) (valuation ρ xs ys v) ↔ s.BottomHeight c
  rw [hu, hv, Scene.bottomHeight_iff wf a']
  have hlevel := (Scene.bottomHeight_iff wf a' u').mp hn
  change s.level c ≤ s.level u' ↔ s.level c = 0
  rw [hlevel]
  omega

theorem wpTable_exact {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (ρ : V → Object α) (a tbl : V) (a' : α)
    (ha : ρ a = some a') (ht : ρ tbl = none) (q : AE V) :
    (wpTable a tbl q).eval (relation s frozen) ρ ↔
      s.Clear a' ∧ q.eval (relation (s.separate a') frozen) ρ := by
  simp only [wpTable, AE.eval_conj, AF.eval_toAE,
    eval_legalSeparate s frozen ρ a tbl a' ha ht, AE.eval_all, AF.eval_implies]
  have rep (ca : s.Clear a') (z : Block.one.Index → Object α)
      (hz : (bottom (.inr ()) (.inl tbl)).eval (relation s frozen) (Sum.elim ρ z)) :
      (rewrittenSeparate (.inl a) (.inl tbl) (.inr ()) (q.map Sum.inl)).eval
        (relation s frozen) (Sum.elim ρ z) ↔ q.eval (relation (s.separate a') frozen) ρ := by
    obtain ⟨u', hu⟩ := bottom_physical s frozen (Sum.elim ρ z) (.inr ()) (.inl tbl) ht hz
    have hn := (eval_bottom s frozen (Sum.elim ρ z) (.inr ()) (.inl tbl) u' hu ht).mp hz
    simpa only [AE.eval_map, Function.comp_def, Sum.elim_inl] using
      rewritten_bottom_correct wf frozen (Sum.elim ρ z) (.inl a) (.inl tbl) (.inr ())
        a' u' ha ht hu ca hn (q.map Sum.inl)
  constructor
  · rintro ⟨ca, h⟩
    have hb : s.BottomHeight (s.root a') := by
      intro t; simp [Scene.Higher, wf.root_zero a']
    have hz := (eval_bottom s frozen (Sum.elim ρ (fun _ : Block.one.Index => some (s.root a')))
      (.inr ()) (.inl tbl) (s.root a') rfl ht).mpr hb
    exact ⟨ca, (rep ca _ hz).mp (h _ hz)⟩
  · rintro ⟨ca, hq⟩
    exact ⟨ca, fun z hz => (rep ca z hz).mpr hq⟩

/-- Public Put dispatches on the target value, including table aliases.
The source must be physical. No source/target distinctness is assumed here;
illegal operands make the calculated WP false. -/
def wp (a b tbl : V) (q : AE V) : AE V :=
  (AF.ofQF (physical a tbl)).toAE.conj
    (((AF.ofQF (.atom .eq b tbl)).implies (wpTable a tbl q)).conj
      ((AF.ofQF (physical b tbl)).implies (wpBlock a b tbl q)))

theorem wp_exact {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (ρ : V → Object α) (a b tbl : V) (ht : ρ tbl = none) (q : AE V) :
    (wp a b tbl q).eval (relation s frozen) ρ ↔
      match ρ a, ρ b with
      | none, _ => False
      | some a', none => s.Clear a' ∧ q.eval (relation (s.separate a') frozen) ρ
      | some a', some b' => (s.Clear a' ∧ s.Clear b' ∧ a' ≠ b') ∧
          q.eval (relation (s.put a' b') frozen) ρ := by
  simp only [wp, AE.eval_conj, AF.eval_toAE, AF.eval_implies, AF.eval_ofQF,
    physical, QF.eval, relation, ht]
  cases ha : ρ a with
  | none => simp
  | some a' =>
    cases hb : ρ b with
    | none => simpa using wpTable_exact wf frozen ρ a tbl a' ha ht q
    | some b' => simpa using wpBlock_exact wf frozen ρ a b tbl a' b' ha hb ht q

/-- This is the previously missing connection to the concrete, full assertion
language rewrite. It is derived from the two execution equivalences. -/
theorem wpBlock_equivWitness {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (ρ : V → Object α) (a b tbl : V) (a' b' : α)
    (ha : ρ a = some a') (hb : ρ b = some b') (ht : ρ tbl = none) (q : AE V) :
    (wpBlock a b tbl q).eval (relation s frozen) ρ ↔
      (WitnessWP.wpPut a' b' (q.toFormula ρ)).eval s frozen := by
  rw [wpBlock_exact wf frozen ρ a b tbl a' b' ha hb ht q, WitnessWP.wpPut_exact wf]
  simp only [Formula.PutStep, AE.eval_toFormula]
  constructor
  · rintro ⟨⟨ca, cb, ne⟩, hq⟩
    exact ⟨s.put a' b', ⟨ca, cb, ne, rfl⟩, hq⟩
  · rintro ⟨s', ⟨ca, cb, ne, rfl⟩, hq⟩
    exact ⟨⟨ca, cb, ne⟩, hq⟩

theorem wpTable_equivWitness {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (ρ : V → Object α) (a tbl : V) (a' : α)
    (ha : ρ a = some a') (ht : ρ tbl = none) (q : AE V) :
    (wpTable a tbl q).eval (relation s frozen) ρ ↔
      (WitnessWP.wpSeparate a' (q.toFormula ρ)).eval s frozen := by
  rw [wpTable_exact wf frozen ρ a tbl a' ha ht q, WitnessWP.wpSeparate_exact wf]
  simp only [Formula.SeparateStep, AE.eval_toFormula]
  constructor
  · rintro ⟨ca, hq⟩
    exact ⟨s.separate a', ⟨ca, rfl⟩, hq⟩
  · rintro ⟨s', ⟨ca, rfl⟩, hq⟩
    exact ⟨ca, hq⟩

/-- Exact nonblocking/demonic WP for the existing operational Put action.
The table constant's interpretation is an explicit environment obligation. -/
theorem wp_action_exact [DecidableEq V] (s : State α V) (wf : s.scene.WellFormed)
    (a b tbl : V) (ht : s.names tbl = none) (q : AE V) :
    (wp a b tbl q).eval (relation s.scene s.frozen) s.names ↔
      (∃ t, (Action.put (.name a) (.name b)).Step s t) ∧
      (∀ t, (Action.put (.name a) (.name b)).Step s t →
        q.eval (relation t.scene t.frozen) t.names) := by
  rw [wp_exact wf s.frozen s.names a b tbl ht q]
  cases ha : s.names a with
  | none => simp [Action.Step, Term.eval, ha]
  | some a' =>
    cases hb : s.names b with
    | none =>
      simp [Action.Step, Term.eval, ha, hb]
      exact fun h => Or.inl h
    | some b' =>
      simp [Action.Step, Term.eval, ha, hb, and_assoc]
      intro ca cb ne
      constructor
      · intro hq t _ _ _ heq; cases heq; exact hq
      · intro h; exact h _ ca cb ne rfl

theorem put_preserves_table [DecidableEq V] (a b tbl : V) (s t : State α V)
    (ht : s.names tbl = none) (run : (Action.put (.name a) (.name b)).Step s t) :
    t.names tbl = none := by
  simp only [Action.Step, Term.eval] at run
  split at run
  · contradiction
  · obtain ⟨_, rfl⟩ := run; exact ht
  · obtain ⟨_, _, _, rfl⟩ := run; exact ht

end Placement
end RoboVerify.Fragments
