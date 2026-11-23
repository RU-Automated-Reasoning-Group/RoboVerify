import RoboVerify.Fragments

set_option autoImplicit false

namespace RoboVerify.Fragments

namespace Block
variable {α : Type}

def allFormula (b : Block) (body : (b.Index → Object α) → Formula α) : Formula α :=
  match b with
  | .empty => body Empty.elim
  | .one => .all fun x => body (fun _ => x)
  | .pair b c => b.allFormula fun x => c.allFormula fun y => body (Sum.elim x y)

theorem eval_allFormula (b : Block) (body : (b.Index → Object α) → Formula α)
    (s frozen : Scene α) :
    (b.allFormula body).eval s frozen ↔ ∀ x, (body x).eval s frozen := by
  induction b with
  | empty =>
    constructor
    · intro h x
      have hx : x = Empty.elim := by funext e; cases e
      simpa only [hx, allFormula] using h
    · intro h; exact h Empty.elim
  | one =>
    change (∀ d, (body (fun _ => d)).eval s frozen) ↔ ∀ x, (body x).eval s frozen
    constructor
    · intro h x
      have hx : (fun _ : Block.one.Index => x ()) = x := by funext e; cases e; rfl
      simpa only [hx] using h (x ())
    · intro h d; exact h (fun _ => d)
  | pair b c ihb ihc =>
    simp only [allFormula, ihb, ihc]
    constructor
    · intro h z
      have hz : Sum.elim (z ∘ Sum.inl) (z ∘ Sum.inr) = z := by
        funext e; cases e <;> rfl
      simpa only [hz] using h (z ∘ Sum.inl) (z ∘ Sum.inr)
    · intro h x y; exact h (Sum.elim x y)

/-- Existential blocks are represented using the logically equivalent
negated-universal form; no object or relation is introduced by this bridge. -/
def exFormula (b : Block) (body : (b.Index → Object α) → Formula α) : Formula α :=
  .neg (b.allFormula fun x => .neg (body x))

theorem eval_exFormula (b : Block) (body : (b.Index → Object α) → Formula α)
    (s frozen : Scene α) :
    (b.exFormula body).eval s frozen ↔ ∃ x, (body x).eval s frozen := by
  classical
  simp [exFormula, Formula.eval, eval_allFormula, Classical.not_forall]

end Block

namespace QF
variable {V α : Type}

def toFormula (ρ : V → Object α) : QF V → Formula α
  | .atom r x y => .atom r (ρ x) (ρ y)
  | .truth => .truth
  | .falsity => .falsity
  | .neg p => .neg (p.toFormula ρ)
  | .conj p q => .conj (p.toFormula ρ) (q.toFormula ρ)
  | .disj p q => .disj (p.toFormula ρ) (q.toFormula ρ)

@[simp] theorem eval_toFormula (p : QF V) (ρ : V → Object α) (s frozen : Scene α) :
    (p.toFormula ρ).eval s frozen ↔ p.eval (relation s frozen) ρ := by
  induction p <;> simp_all [toFormula, Formula.eval, eval]

end QF

namespace AE
variable {V α : Type}

def toFormula (p : AE V) (ρ : V → Object α) : Formula α :=
  p.u.allFormula fun x => p.e.exFormula fun y => p.matrix.toFormula (valuation ρ x y)

@[simp] theorem eval_toFormula (p : AE V) (ρ : V → Object α) (s frozen : Scene α) :
    (p.toFormula ρ).eval s frozen ↔ p.eval (relation s frozen) ρ := by
  simp only [toFormula, Block.eval_allFormula, Block.eval_exFormula, QF.eval_toFormula, eval]

end AE
end RoboVerify.Fragments
