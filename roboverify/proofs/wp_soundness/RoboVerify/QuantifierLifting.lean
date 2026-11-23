import Std

set_option autoImplicit false
set_option warningAsError true

namespace RoboVerify.QuantifierLifting

/-- Nonblocking demonic WP for a fixed-domain transition relation. -/
def must {State : Type} (step : State → State → Prop)
    (q : State → Prop) (s : State) : Prop :=
  (∃ t, step s t) ∧ ∀ t, step s t → q t

/-- The logical variable ranges over the same nonempty domain before and after
the action. This is the semantic step used to lift AFR closure to AER closure. -/
theorem must_forall {State Obj : Type} [Nonempty Obj]
    (step : State → State → Prop) (q : Obj → State → Prop) (s : State) :
    must step (fun t => ∀ x, q x t) s ↔ ∀ x, must step (q x) s := by
  constructor
  · rintro ⟨enabled, correct⟩ x
    exact ⟨enabled, fun t ht => correct t ht x⟩
  · intro h
    obtain ⟨x⟩ := ‹Nonempty Obj›
    exact ⟨(h x).1, fun t ht y => (h y).2 t ht⟩

/-- A sound Get includes existence and correctness for every eligible witness.
This identity displays its forall-exists normal form when g and q are
quantifier-free matrices. The theorem checks the logical identity, not syntax. -/
theorem get_normal_form {Choice Univ Ex : Type}
    [Nonempty Choice] [Nonempty Univ] [Nonempty Ex]
    (g : Choice → Prop) (q : Choice → Univ → Ex → Prop) :
    ((∃ w, g w) ∧ ∀ a, g a → ∀ u, ∃ v, q a u v) ↔
      ∀ a u, ∃ w v, g w ∧ (g a → q a u v) := by
  classical
  constructor
  · rintro ⟨⟨w, hw⟩, correct⟩ a u
    by_cases ha : g a
    · obtain ⟨v, hv⟩ := correct a ha u
      exact ⟨w, v, hw, fun _ => hv⟩
    · obtain ⟨v⟩ := ‹Nonempty Ex›
      exact ⟨w, v, hw, fun h => (ha h).elim⟩
  · intro h
    constructor
    · obtain ⟨a⟩ := ‹Nonempty Choice›
      obtain ⟨u⟩ := ‹Nonempty Univ›
      obtain ⟨w, _, hw, _⟩ := h a u
      exact ⟨w, hw⟩
    · intro a ha u
      obtain ⟨_, v, _, hv⟩ := h a u
      exact ⟨v, hv ha⟩

/-- The no-witness exit case with a universal invariant and postcondition
has an exists-forall counterexample formula when i, g, and q are QF. -/
theorem exit_refutation_normal_form {Inv Guard Post : Type}
    (i : Inv → Prop) (g : Guard → Prop) (q : Post → Prop) :
    ¬ ((∀ x, i x) ∧ ¬ (∃ v, g v) → ∀ y, q y) ↔
      ∃ y, (∀ x, i x) ∧ (∀ v, ¬ g v) ∧ ¬ q y := by
  classical
  simp only [Classical.not_imp, Classical.not_forall, not_exists]
  constructor
  · rintro ⟨⟨hi, hg⟩, y, hy⟩
    exact ⟨y, hi, hg, hy⟩
  · rintro ⟨y, hi, hg, hy⟩
    exact ⟨⟨hi, hg⟩, y, hy⟩

/-- Guarded implication alone incorrectly accepts an empty standalone Get. -/
theorem get_existence_is_necessary :
    (∀ x : Bool, False → x = true) ∧
      ¬((∃ _ : Bool, False) ∧ ∀ x : Bool, False → x = true) := by
  simp

/-- If true is eligible and false is not, the else branch is unreachable.
Checking the else branch separately for each ineligible value is too strong. -/
theorem per_witness_else_is_wrong :
    ((∀ x : Bool, x = true → True) ∧ ((∀ x : Bool, x ≠ true) → False)) ∧
      ¬((∀ x : Bool, x = true → True) ∧ (∀ x : Bool, x ≠ true → False)) := by
  simp

end RoboVerify.QuantifierLifting
