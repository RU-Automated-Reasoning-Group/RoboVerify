import RoboVerify.Formula

set_option autoImplicit false

namespace RoboVerify.Fragments

/-- A finite block of first-order variables. A pair concatenates blocks;
it does not introduce an object-valued function or a relation variable. -/
inductive Block where
  | empty | one | pair : Block → Block → Block

def Block.Index : Block → Type
  | .empty => Empty
  | .one => Unit
  | .pair a b => a.Index ⊕ b.Index

/-- A separate, deep syntax makes the function-free restriction explicit.
Variables include free parameters/constants; terms have no function symbols. -/
inductive QF (V : Type) where
  | atom : Relation → V → V → QF V
  | truth | falsity
  | neg : QF V → QF V
  | conj : QF V → QF V → QF V
  | disj : QF V → QF V → QF V

namespace QF
variable {V W D : Type}

def eval (M : Relation → D → D → Prop) (ρ : V → D) : QF V → Prop
  | .atom r x y => M r (ρ x) (ρ y)
  | .truth => True
  | .falsity => False
  | .neg p => ¬p.eval M ρ
  | .conj p q => p.eval M ρ ∧ q.eval M ρ
  | .disj p q => p.eval M ρ ∨ q.eval M ρ

def map (f : V → W) : QF V → QF W
  | .atom r x y => .atom r (f x) (f y)
  | .truth => .truth
  | .falsity => .falsity
  | .neg p => .neg (p.map f)
  | .conj p q => .conj (p.map f) (q.map f)
  | .disj p q => .disj (p.map f) (q.map f)

@[simp] theorem eval_map (f : V → W) (M : Relation → D → D → Prop)
    (ρ : W → D) (q : QF V) : (q.map f).eval M ρ ↔ q.eval M (ρ ∘ f) := by
  induction q <;> simp_all [map, eval]

/-- Simultaneous substitution of QF formulas for atoms. -/
def replace (f : Relation → V → V → QF V) : QF V → QF V
  | .atom r x y => f r x y
  | .truth => .truth
  | .falsity => .falsity
  | .neg p => .neg (p.replace f)
  | .conj p q => .conj (p.replace f) (q.replace f)
  | .disj p q => .disj (p.replace f) (q.replace f)

theorem eval_replace (f : Relation → V → V → QF V)
    (M N : Relation → D → D → Prop) (ρ : V → D)
    (correct : ∀ r x y, (f r x y).eval M ρ ↔ N r (ρ x) (ρ y)) (q : QF V) :
    (q.replace f).eval M ρ ↔ q.eval N ρ := by
  induction q <;> simp_all [replace, eval]

end QF

abbrev Vars (V : Type) (u e : Block) := V ⊕ (u.Index ⊕ e.Index)

def valuation {V D : Type} {u e : Block}
    (ρ : V → D) (x : u.Index → D) (y : e.Index → D) : Vars V u e → D :=
  Sum.elim ρ (Sum.elim x y)

/-- An actual finite forall-exists prenex form with a function-free matrix. -/
structure AE (V : Type) where
  u : Block
  e : Block
  matrix : QF (Vars V u e)

/-- EPR has the opposite prefix; empty blocks are permitted. -/
structure EA (V : Type) where
  e : Block
  u : Block
  matrix : QF (Vars V u e)

def AE.eval {V D : Type} (p : AE V) (M : Relation → D → D → Prop)
    (ρ : V → D) : Prop :=
  ∀ x, ∃ y, p.matrix.eval M (valuation ρ x y)

def EA.eval {V D : Type} (p : EA V) (M : Relation → D → D → Prop)
    (ρ : V → D) : Prop :=
  ∃ y, ∀ x, p.matrix.eval M (valuation ρ x y)

namespace AE
variable {V W D : Type}

def ofQF (p : QF V) : AE V := ⟨.empty, .empty, p.map Sum.inl⟩

@[simp] theorem eval_ofQF (p : QF V) (M : Relation → D → D → Prop) (ρ : V → D) :
    (ofQF p).eval M ρ ↔ p.eval M ρ := by
  simp only [ofQF, eval, QF.eval_map]
  constructor
  · intro h
    exact (h Empty.elim).choose_spec
  · intro h _
    exact ⟨Empty.elim, h⟩

def map (f : V → W) (p : AE V) : AE W :=
  ⟨p.u, p.e, p.matrix.map (Sum.map f id)⟩

/-- QF atom replacement keeps the same finite quantifier prefix. -/
def replace (p : AE V)
    (f : Relation → Vars V p.u p.e → Vars V p.u p.e → QF (Vars V p.u p.e)) : AE V :=
  ⟨p.u, p.e, p.matrix.replace f⟩

theorem eval_replace (p : AE V)
    (f : Relation → Vars V p.u p.e → Vars V p.u p.e → QF (Vars V p.u p.e))
    (M N : Relation → D → D → Prop) (ρ : V → D)
    (correct : ∀ x y r v w, (f r v w).eval M (valuation ρ x y) ↔
      N r (valuation ρ x y v) (valuation ρ x y w)) :
    (p.replace f).eval M ρ ↔ p.eval N ρ := by
  simp only [replace, eval]
  apply forall_congr'; intro x
  apply exists_congr; intro y
  exact QF.eval_replace f M N _ (correct x y) p.matrix

@[simp] theorem eval_map (f : V → W) (p : AE V)
    (M : Relation → D → D → Prop) (ρ : W → D) :
    (p.map f).eval M ρ ↔ p.eval M (ρ ∘ f) := by
  simp only [map, eval, QF.eval_map]
  apply forall_congr'; intro x
  apply exists_congr; intro y
  have h : valuation ρ x y ∘ Sum.map f id = valuation (ρ ∘ f) x y := by
    funext v; cases v <;> rfl
  rw [h]

def left {u e u' e' : Block} : Vars V u e → Vars V (.pair u u') (.pair e e')
  | .inl v => .inl v
  | .inr (.inl x) => .inr (.inl (.inl x))
  | .inr (.inr y) => .inr (.inr (.inl y))

def right {u e u' e' : Block} : Vars V u' e' → Vars V (.pair u u') (.pair e e')
  | .inl v => .inl v
  | .inr (.inl x) => .inr (.inl (.inr x))
  | .inr (.inr y) => .inr (.inr (.inr y))

@[simp] theorem valuation_left {u e u' e' : Block}
    (ρ : V → D) (x : (Block.pair u u').Index → D) (y : (Block.pair e e').Index → D) :
    valuation ρ x y ∘ left = valuation ρ (x ∘ Sum.inl) (y ∘ Sum.inl) := by
  funext v; cases v with
  | inl => rfl
  | inr v => cases v <;> rfl

@[simp] theorem valuation_right {u e u' e' : Block}
    (ρ : V → D) (x : (Block.pair u u').Index → D) (y : (Block.pair e e').Index → D) :
    valuation ρ x y ∘ right = valuation ρ (x ∘ Sum.inr) (y ∘ Sum.inr) := by
  funext v; cases v with
  | inl => rfl
  | inr v => cases v <;> rfl

def conj (p q : AE V) : AE V :=
  ⟨.pair p.u q.u, .pair p.e q.e,
    .conj (p.matrix.map left) (q.matrix.map right)⟩

def disj (p q : AE V) : AE V :=
  ⟨.pair p.u q.u, .pair p.e q.e,
    .disj (p.matrix.map left) (q.matrix.map right)⟩

theorem eval_conj [Nonempty D] (p q : AE V)
    (M : Relation → D → D → Prop) (ρ : V → D) :
    (p.conj q).eval M ρ ↔ p.eval M ρ ∧ q.eval M ρ := by
  classical
  simp only [conj, eval, QF.eval, QF.eval_map, valuation_left, valuation_right]
  constructor
  · intro h
    obtain ⟨d⟩ := ‹Nonempty D›
    constructor
    · intro x
      obtain ⟨y, hp, _⟩ := h (Sum.elim x (fun _ => d))
      exact ⟨y ∘ Sum.inl, hp⟩
    · intro x
      obtain ⟨y, _, hq⟩ := h (Sum.elim (fun _ => d) x)
      exact ⟨y ∘ Sum.inr, hq⟩
  · rintro ⟨hp, hq⟩ x
    obtain ⟨y, hy⟩ := hp (x ∘ Sum.inl)
    obtain ⟨z, hz⟩ := hq (x ∘ Sum.inr)
    exact ⟨Sum.elim y z, hy, hz⟩

theorem eval_disj [Nonempty D] (p q : AE V)
    (M : Relation → D → D → Prop) (ρ : V → D) :
    (p.disj q).eval M ρ ↔ p.eval M ρ ∨ q.eval M ρ := by
  classical
  simp only [disj, eval, QF.eval, QF.eval_map, valuation_left, valuation_right]
  obtain ⟨d⟩ := ‹Nonempty D›
  constructor
  · intro h
    by_cases hp : ∀ x, ∃ y, p.matrix.eval M (valuation ρ x y)
    · exact Or.inl hp
    · right
      obtain ⟨x, hx⟩ := Classical.not_forall.mp hp
      intro z
      obtain ⟨y, hy | hy⟩ := h (Sum.elim x z)
      · exact (hx ⟨y ∘ Sum.inl, hy⟩).elim
      · exact ⟨y ∘ Sum.inr, hy⟩
  · intro h x
    rcases h with hp | hq
    · obtain ⟨y, hy⟩ := hp (x ∘ Sum.inl)
      exact ⟨Sum.elim y (fun _ => d), Or.inl hy⟩
    · obtain ⟨y, hy⟩ := hq (x ∘ Sum.inr)
      exact ⟨Sum.elim (fun _ => d) y, Or.inr hy⟩

def all (b : Block) (p : AE (V ⊕ b.Index)) : AE V :=
  ⟨.pair b p.u, p.e, p.matrix.map fun
    | .inl (.inl v) => .inl v
    | .inl (.inr z) => .inr (.inl (.inl z))
    | .inr (.inl x) => .inr (.inl (.inr x))
    | .inr (.inr y) => .inr (.inr y)⟩

theorem eval_all (b : Block) (p : AE (V ⊕ b.Index))
    (M : Relation → D → D → Prop) (ρ : V → D) :
    (all b p).eval M ρ ↔ ∀ z, p.eval M (Sum.elim ρ z) := by
  simp only [all, eval, QF.eval_map]
  have eq (x : (Block.pair b p.u).Index → D) (y : p.e.Index → D) :
      valuation ρ x y ∘ (fun
        | .inl (.inl v) => .inl v
        | .inl (.inr z) => .inr (.inl (.inl z))
        | .inr (.inl u) => .inr (.inl (.inr u))
        | .inr (.inr e) => .inr (.inr e)) =
      valuation (Sum.elim ρ (x ∘ Sum.inl)) (x ∘ Sum.inr) y := by
    funext v; cases v with
    | inl v => cases v <;> rfl
    | inr v => cases v <;> rfl
  simp only [eq]
  constructor
  · intro h z x
    exact h (Sum.elim z x)
  · intro h x
    exact h (x ∘ Sum.inl) (x ∘ Sum.inr)

def refute (p : AE V) : EA V :=
  ⟨p.u, p.e, .neg (p.matrix.map (Sum.map id Sum.swap))⟩

theorem eval_refute (p : AE V) (M : Relation → D → D → Prop) (ρ : V → D) :
    p.refute.eval M ρ ↔ ¬p.eval M ρ := by
  classical
  simp only [refute, EA.eval, eval, QF.eval, QF.eval_map,
    Classical.not_forall, not_exists]
  apply exists_congr; intro y
  apply forall_congr'; intro x
  have h : valuation ρ x y ∘ Sum.map id Sum.swap = valuation ρ y x := by
    funext v; cases v with
    | inl => rfl
    | inr v => cases v <;> rfl
  rw [h]

end AE

/-- Alternation-free assertions: Boolean combinations of single universal
or existential blocks over QF matrices. No nested alternation is admitted. -/
inductive AF (V : Type) where
  | all : (b : Block) → QF (V ⊕ b.Index) → AF V
  | ex : (b : Block) → QF (V ⊕ b.Index) → AF V
  | conj : AF V → AF V → AF V
  | disj : AF V → AF V → AF V

namespace AF
variable {V D : Type}

def eval (M : Relation → D → D → Prop) (ρ : V → D) : AF V → Prop
  | .all _ p => ∀ x, p.eval M (Sum.elim ρ x)
  | .ex _ p => ∃ x, p.eval M (Sum.elim ρ x)
  | .conj p q => p.eval M ρ ∧ q.eval M ρ
  | .disj p q => p.eval M ρ ∨ q.eval M ρ

def neg : AF V → AF V
  | .all b p => .ex b (.neg p)
  | .ex b p => .all b (.neg p)
  | .conj p q => .disj p.neg q.neg
  | .disj p q => .conj p.neg q.neg

theorem eval_neg (p : AF V) (M : Relation → D → D → Prop) (ρ : V → D) :
    p.neg.eval M ρ ↔ ¬p.eval M ρ := by
  classical
  induction p with
  | all b p => simp [neg, eval, QF.eval, Classical.not_forall]
  | ex b p => simp [neg, eval, QF.eval]
  | conj p q hp hq =>
    simp only [neg, eval, hp, hq]
    exact Classical.not_and_iff_not_or_not.symm
  | disj p q hp hq => simp [neg, eval, hp, hq]

def toAE : AF V → AE V
  | .all b p => AE.all b (AE.ofQF p)
  | .ex b p => ⟨.empty, b, p.map fun
      | .inl v => .inl v
      | .inr y => .inr (.inr y)⟩
  | .conj p q => p.toAE.conj q.toAE
  | .disj p q => p.toAE.disj q.toAE

theorem eval_toAE [Nonempty D] (p : AF V) (M : Relation → D → D → Prop)
    (ρ : V → D) : p.toAE.eval M ρ ↔ p.eval M ρ := by
  induction p with
  | all b p => simp [toAE, AE.eval_all, eval]
  | ex b p =>
    simp only [toAE, AE.eval, eval, QF.eval_map]
    have eq (x : Block.empty.Index → D) (y : b.Index → D) :
        valuation ρ x y ∘ (fun
          | .inl v => .inl v
          | .inr e => .inr (.inr e)) = Sum.elim ρ y := by
      funext v; cases v <;> rfl
    simp only [eq]
    constructor
    · intro h; exact h Empty.elim
    · intro h _; exact h
  | conj p q hp hq => simp [toAE, AE.eval_conj, hp, hq, eval]
  | disj p q hp hq => simp [toAE, AE.eval_disj, hp, hq, eval]

def implies (p : AF V) (q : AE V) : AE V := p.neg.toAE.disj q

theorem eval_implies [Nonempty D] (p : AF V) (q : AE V)
    (M : Relation → D → D → Prop) (ρ : V → D) :
    (p.implies q).eval M ρ ↔ (p.eval M ρ → q.eval M ρ) := by
  classical
  simp only [implies, AE.eval_disj, eval_toAE, eval_neg]
  by_cases h : p.eval M ρ <;> simp [h]

end AF

namespace EA
variable {V D : Type}

/-- Existentially close a finite block of free program parameters. This does
not enumerate the object domain: each slot still ranges over all objects. -/
def close (b : Block) (p : EA b.Index) : EA Empty :=
  ⟨.pair b p.e, p.u, p.matrix.map fun
    | .inl v => .inr (.inr (.inl v))
    | .inr (.inl x) => .inr (.inl x)
    | .inr (.inr y) => .inr (.inr (.inr y))⟩

theorem eval_close (b : Block) (p : EA b.Index) (M : Relation → D → D → Prop) :
    (close b p).eval M Empty.elim ↔ ∃ ρ, p.eval M ρ := by
  simp only [close, eval, QF.eval_map]
  have eq (y : (Block.pair b p.e).Index → D) (x : p.u.Index → D) :
      valuation Empty.elim x y ∘ (fun
        | .inl v => .inr (.inr (.inl v))
        | .inr (.inl u) => .inr (.inl u)
        | .inr (.inr e) => .inr (.inr (.inr e))) =
      valuation (y ∘ Sum.inl) x (y ∘ Sum.inr) := by
    funext v; cases v with
    | inl => rfl
    | inr v => cases v <;> rfl
  simp only [eq]
  constructor
  · rintro ⟨y, h⟩
    exact ⟨y ∘ Sum.inl, y ∘ Sum.inr, h⟩
  · rintro ⟨ρ, y, h⟩
    exact ⟨Sum.elim ρ y, h⟩

def negate (p : EA V) : AE V :=
  ⟨p.e, p.u, .neg (p.matrix.map (Sum.map id Sum.swap))⟩

theorem eval_negate (p : EA V) (M : Relation → D → D → Prop) (ρ : V → D) :
    p.negate.eval M ρ ↔ ¬p.eval M ρ := by
  classical
  simp only [negate, AE.eval, eval, QF.eval, QF.eval_map,
    not_exists, Classical.not_forall]
  apply forall_congr'; intro y
  apply exists_congr; intro x
  have h : valuation ρ y x ∘ Sum.map id Sum.swap = valuation ρ x y := by
    funext v; cases v with
    | inl => rfl
    | inr v => cases v <;> rfl
  rw [h]

/-- Background axioms occur positively in the solver query. This construction
requires their EPR representation; arbitrary forall-exists axioms do not fit. -/
def counterexample (background : EA V) (vc : AE V) : EA V :=
  (background.negate.disj vc).refute

theorem eval_counterexample [Nonempty D] (background : EA V) (vc : AE V)
    (M : Relation → D → D → Prop) (ρ : V → D) :
    (background.counterexample vc).eval M ρ ↔
      background.eval M ρ ∧ ¬vc.eval M ρ := by
  classical
  simp [counterexample, AE.eval_refute, AE.eval_disj, eval_negate]

end EA
end RoboVerify.Fragments
