import RoboVerify.Fragments
import RoboVerify.QuantifierLifting

set_option autoImplicit false

namespace RoboVerify.Fragments

namespace AF
variable {V W D : Type}

def ofQF (p : QF V) : AF V := .all .empty (p.map Sum.inl)

@[simp] theorem eval_ofQF (p : QF V) (M : Relation → D → D → Prop) (ρ : V → D) :
    (ofQF p).eval M ρ ↔ p.eval M ρ := by
  simp only [ofQF, eval, QF.eval_map]
  exact ⟨fun h => h Empty.elim, fun h _ => h⟩

def map (f : V → W) : AF V → AF W
  | .all b p => .all b (p.map (Sum.map f id))
  | .ex b p => .ex b (p.map (Sum.map f id))
  | .conj p q => .conj (p.map f) (q.map f)
  | .disj p q => .disj (p.map f) (q.map f)

@[simp] theorem eval_map (f : V → W) (p : AF V)
    (M : Relation → D → D → Prop) (ρ : W → D) :
    (p.map f).eval M ρ ↔ p.eval M (ρ ∘ f) := by
  have eq {b : Block} (x : b.Index → D) :
      Sum.elim ρ x ∘ Sum.map f id = Sum.elim (ρ ∘ f) x := by
    funext v; cases v <;> rfl
  induction p <;> simp_all [map, eval]

end AF

/-- Uniform AFR preservation for the existential suffix suffices for AER
preservation. Logical parameters x are fixed across the action; the object
domain is fixed and nonempty. The AFR result is an explicit hypothesis, not
an axiom claiming that current Put or arbitrary actions preserve AFR. -/
theorem lift_AF_action {V D S : Type} [Nonempty D]
    (step : S → S → Prop) (M : S → Relation → D → D → Prop)
    (ρ : S → V → D) (q : AE V) (result : AF (V ⊕ q.u.Index))
    (correct : ∀ s x, result.eval (M s) (Sum.elim (ρ s) x) ↔
      QuantifierLifting.must step
        (fun t => ∃ y, q.matrix.eval (M t) (valuation (ρ t) x y)) s) (s : S) :
    (AE.all q.u result.toAE).eval (M s) (ρ s) ↔
      QuantifierLifting.must step (fun t => q.eval (M t) (ρ t)) s := by
  rw [AE.eval_all]
  simp only [AF.eval_toAE, correct]
  exact (QuantifierLifting.must_forall step _ s).symm

namespace Rules
variable {V D : Type}

/-- The normal form of the shared-height-witness WP. QF atom replacements
produce qMax and qNext using AE.replace; selectors and legality are AF.
Consequently this construction preserves AER directly, without requiring
the stronger and unproved claim that Put preserves AFR. -/
def sharedPut (b : Block) (legal maximum : AF V) (next : AF (V ⊕ b.Index))
    (qMax : AE V) (qNext : AE (V ⊕ b.Index)) : AE V :=
  legal.toAE.conj ((maximum.implies qMax).conj (AE.all b (next.implies qNext)))

theorem sharedPut_exact [Nonempty D] (b : Block) (legal maximum : AF V)
    (next : AF (V ⊕ b.Index)) (qMax : AE V) (qNext : AE (V ⊕ b.Index))
    (M : Relation → D → D → Prop) (ρ : V → D) (post : Prop)
    (coverage : maximum.eval M ρ ∨ ∃ x, next.eval M (Sum.elim ρ x))
    (maxCorrect : maximum.eval M ρ → (qMax.eval M ρ ↔ post))
    (nextCorrect : ∀ x, next.eval M (Sum.elim ρ x) →
      (qNext.eval M (Sum.elim ρ x) ↔ post)) :
    (sharedPut b legal maximum next qMax qNext).eval M ρ ↔
      legal.eval M ρ ∧ post := by
  simp only [sharedPut, AE.eval_conj, AF.eval_toAE, AF.eval_implies, AE.eval_all]
  constructor
  · rintro ⟨hl, hm, hn⟩
    refine ⟨hl, ?_⟩
    rcases coverage with h | ⟨x, h⟩
    · exact (maxCorrect h).mp (hm h)
    · exact (nextCorrect x h).mp (hn x h)
  · rintro ⟨hl, hp⟩
    exact ⟨hl, fun h => (maxCorrect h).mpr hp, fun x h => (nextCorrect x h).mpr hp⟩

/-- bind maps each old program name to its selected witness or its old value.
It is a variable substitution, never an object-valued function symbol. -/
def selectAll (b : Block) (g : QF (V ⊕ b.Index)) (bind : V → V ⊕ b.Index)
    (q : AE V) : AE V := AE.all b ((AF.ofQF g).implies (q.map bind))

def get (b : Block) (g : QF (V ⊕ b.Index)) (bind : V → V ⊕ b.Index)
    (q : AE V) : AE V := (AF.ex b g).toAE.conj (selectAll b g bind q)

def chooseIf (b : Block) (g : QF (V ⊕ b.Index)) (bind : V → V ⊕ b.Index)
    (yes no : AE V) : AE V :=
  (selectAll b g bind yes).conj ((AF.all b (.neg g)).implies no)

theorem eval_selectAll [Nonempty D] (b : Block) (g : QF (V ⊕ b.Index))
    (bind : V → V ⊕ b.Index) (q : AE V) (M : Relation → D → D → Prop) (ρ : V → D) :
    (selectAll b g bind q).eval M ρ ↔
      ∀ x, g.eval M (Sum.elim ρ x) → q.eval M (Sum.elim ρ x ∘ bind) := by
  simp [selectAll, AE.eval_all, AF.eval_implies]

theorem eval_get [Nonempty D] (b : Block) (g : QF (V ⊕ b.Index))
    (bind : V → V ⊕ b.Index) (q : AE V) (M : Relation → D → D → Prop) (ρ : V → D) :
    (get b g bind q).eval M ρ ↔
      (∃ x, g.eval M (Sum.elim ρ x)) ∧
      (∀ x, g.eval M (Sum.elim ρ x) → q.eval M (Sum.elim ρ x ∘ bind)) := by
  simp only [get, AE.eval_conj, AF.eval_toAE, AF.eval, eval_selectAll]

theorem eval_chooseIf [Nonempty D] (b : Block) (g : QF (V ⊕ b.Index))
    (bind : V → V ⊕ b.Index) (yes no : AE V)
    (M : Relation → D → D → Prop) (ρ : V → D) :
    (chooseIf b g bind yes no).eval M ρ ↔
      (∀ x, g.eval M (Sum.elim ρ x) → yes.eval M (Sum.elim ρ x ∘ bind)) ∧
      ((∀ x, ¬g.eval M (Sum.elim ρ x)) → no.eval M ρ) := by
  simp only [chooseIf, AE.eval_conj, eval_selectAll, AF.eval_implies, AF.eval, QF.eval]

def preserve (b : Block) (g : QF (V ⊕ b.Index)) (bind : V → V ⊕ b.Index)
    (inv : AF V) (body : AE V) : AE V :=
  AE.all b ((AF.conj (inv.map Sum.inl) (AF.ofQF g)).implies (body.map bind))

def exit (b : Block) (g : QF (V ⊕ b.Index)) (inv : AF V) (q : AE V) : AE V :=
  (AF.conj inv (.all b (.neg g))).implies q

theorem eval_preserve [Nonempty D] (b : Block) (g : QF (V ⊕ b.Index))
    (bind : V → V ⊕ b.Index) (inv : AF V) (body : AE V)
    (M : Relation → D → D → Prop) (ρ : V → D) :
    (preserve b g bind inv body).eval M ρ ↔
      ∀ x, inv.eval M ρ ∧ g.eval M (Sum.elim ρ x) →
        body.eval M (Sum.elim ρ x ∘ bind) := by
  simp [preserve, AE.eval_all, AF.eval_implies, AF.eval]

theorem eval_exit [Nonempty D] (b : Block) (g : QF (V ⊕ b.Index))
    (inv : AF V) (q : AE V) (M : Relation → D → D → Prop) (ρ : V → D) :
    (exit b g inv q).eval M ρ ↔
      (inv.eval M ρ ∧ (∀ x, ¬g.eval M (Sum.elim ρ x)) → q.eval M ρ) := by
  simp only [exit, AF.eval_implies, AF.eval, QF.eval]

end Rules

/-- The restricted symbolic VC generator. Each abstract action must supply an
AER transformer. Its semantic correctness is a separate premise: membership
alone never certifies an action. lift_AF_action explains the proposed weaker
uniform-AFR hypothesis. Guards here are QF matrices, including tuple choices. -/
inductive Program (V : Type) where
  | skip : Program V
  | action : (AE V → AE V) → Program V
  | get : (b : Block) → QF (V ⊕ b.Index) → (V → V ⊕ b.Index) → Program V
  | seq : Program V → Program V → Program V
  | branch : (b : Block) → QF (V ⊕ b.Index) → (V → V ⊕ b.Index) →
      Program V → Program V → Program V
  | loop : (b : Block) → QF (V ⊕ b.Index) → (V → V ⊕ b.Index) →
      AF V → Program V → Program V

namespace Program
variable {V D : Type}

def wp : Program V → AE V → AE V
  | .skip, q => q
  | .action transform, q => transform q
  | .get b g bind, q => Rules.get b g bind q
  | .seq c d, q => c.wp (d.wp q)
  | .branch b g bind c d, q => Rules.chooseIf b g bind (c.wp q) (d.wp q)
  | .loop _ _ _ inv _, _ => inv.toAE

def vcs : Program V → AE V → List (AE V)
  | .skip, _ | .action _, _ | .get _ _ _, _ => []
  | .seq c d, q => c.vcs (d.wp q) ++ d.vcs q
  | .branch _ _ _ c d, q => c.vcs q ++ d.vcs q
  | .loop b g bind inv body, q =>
    Rules.preserve b g bind inv (body.wp inv.toAE) ::
      Rules.exit b g inv q :: body.vcs inv.toAE

def vcgen (p : AF V) (c : Program V) (q : AF V) : List (AE V) :=
  p.implies (c.wp q.toAE) :: c.vcs q.toAE

/-- Every VC of this normalizing generator has an equivalent EPR refutation.
No validity, domain-axiom encoding, or production Python correspondence is
assumed by this syntactic theorem. -/
theorem vcgen_epr (p q : AF V) (c : Program V) :
    ∀ vc ∈ vcgen p c q, ∃ refutation : EA V,
      ∀ (D : Type) (M : Relation → D → D → Prop) (ρ : V → D),
        refutation.eval M ρ ↔ ¬vc.eval M ρ := by
  intro vc _
  exact ⟨vc.refute, fun _ M ρ => AE.eval_refute vc M ρ⟩

theorem vcgen_background_epr (p q : AF V) (c : Program V) (background : EA V) :
    ∀ vc ∈ vcgen p c q, ∃ query : EA V,
      ∀ (D : Type) [Nonempty D] (M : Relation → D → D → Prop) (ρ : V → D),
        query.eval M ρ ↔ background.eval M ρ ∧ ¬vc.eval M ρ := by
  intro vc _
  exact ⟨background.counterexample vc, fun _ _ M ρ => EA.eval_counterexample background vc M ρ⟩

end Program
end RoboVerify.Fragments
