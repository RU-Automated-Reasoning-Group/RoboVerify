import RoboVerify.Effects

set_option autoImplicit false

namespace RoboVerify

/-- none is the relational table constant; some x is a physical block. -/
abbrev Object (α : Type) := Option α

inductive Relation where
  | on | higher | scattered | onZero | eq
  deriving DecidableEq, Repr

/-- Quantifiers bind object values using Lean functions. This representation
makes variable capture impossible; it does not certify Python's AST traversal. -/
inductive Formula (α : Type) where
  | atom : Relation → Object α → Object α → Formula α
  | truth : Formula α
  | falsity : Formula α
  | neg : Formula α → Formula α
  | conj : Formula α → Formula α → Formula α
  | disj : Formula α → Formula α → Formula α
  | all : (Object α → Formula α) → Formula α
  | ex : (Object α → Formula α) → Formula α

def relation {α : Type} (s frozen : Scene α) (r : Relation) (x y : Object α) : Prop :=
  match r with
  | .eq => x = y
  | .onZero => match x, y with
    | some x, some y => frozen.On x y
    | none, none => True
    | _, _ => False
  | .on => match x, y with
    | some x, some y => s.On x y
    | none, none => True
    | _, _ => False
  | .higher => match x, y with
    | some x, some y => s.Higher x y
    | none, none => True
    | _, _ => False
  | .scattered => match x, y with
    | some x, some y => s.Scattered x y
    | _, _ => False

namespace Formula
variable {α : Type}

def eval (s frozen : Scene α) : Formula α → Prop
  | .atom r x y => relation s frozen r x y
  | .truth => True
  | .falsity => False
  | .neg p => ¬ p.eval s frozen
  | .conj p q => p.eval s frozen ∧ q.eval s frozen
  | .disj p q => p.eval s frozen ∨ q.eval s frozen
  | .all body => ∀ x, (body x).eval s frozen
  | .ex body => ∃ x, (body x).eval s frozen

def imp (p q : Formula α) : Formula α := .disj (.neg p) q

def iteF (p q r : Formula α) : Formula α :=
  .disj (.conj p q) (.conj (.neg p) r)

def phys (r : Relation) (x y : α) : Formula α := .atom r (some x) (some y)

def allPhysical (body : α → Formula α) : Formula α :=
  .all fun x => match x with
    | none => .truth
    | some x => body x

@[simp] theorem eval_allPhysical (body : α → Formula α) (s frozen : Scene α) :
    (allPhysical body).eval s frozen ↔ ∀ x, (body x).eval s frozen := by
  simp [allPhysical, eval, Option.forall]

@[simp] theorem eval_imp (p q : Formula α) (s frozen : Scene α) :
    (imp p q).eval s frozen ↔ (p.eval s frozen → q.eval s frozen) := by
  classical
  by_cases hp : p.eval s frozen <;> simp [imp, eval, hp]

def nextHeight (b c : α) : Formula α :=
  .disj (phys .higher b c)
    (.conj (phys .higher c b)
      (allPhysical fun t =>
        imp (.conj (phys .higher c t) (.neg (phys .higher t c)))
          (phys .higher b t)))

def bottomHeight (c : α) : Formula α :=
  allPhysical fun t => phys .higher t c

def clear (a : α) : Formula α :=
  allPhysical fun x => imp (phys .on x a) (phys .eq x a)

@[simp] theorem eval_nextHeight (s frozen : Scene α) (b c : α) :
    (nextHeight b c).eval s frozen ↔ s.NextHeight b c := by
  simp only [nextHeight, eval, eval_allPhysical, eval_imp, phys, relation,
    Scene.NextHeight]

@[simp] theorem eval_bottomHeight (s frozen : Scene α) (c : α) :
    (bottomHeight c).eval s frozen ↔ s.BottomHeight c := by
  simp [bottomHeight, phys, eval, relation, allPhysical, Scene.BottomHeight, Option.forall]

@[simp] theorem eval_clear (s frozen : Scene α) (a : α) :
    (clear a).eval s frozen ↔ s.Clear a := by
  simp only [clear, eval_allPhysical, eval_imp, phys, eval, relation,
    Option.some.injEq, Scene.Clear]

/-- Formulas below use equality tests in the assertion language, not a solver
or meta-level decision about whether symbolic operands are equal. -/
def putPhysical (a b : α) (r : Relation) (x y : α) : Formula α :=
  match r with
  | .on =>
    .disj (.conj (.neg (phys .eq x a)) (phys .on x y))
      (.conj (phys .eq x a) (.disj (phys .eq y a) (phys .on b y)))
  | .higher =>
    iteF (phys .eq x a) (.disj (phys .eq y a) (nextHeight b y))
      (iteF (phys .eq y a)
        (.conj (phys .higher x b) (.neg (phys .higher b x)))
        (phys .higher x y))
  | .scattered =>
    iteF (phys .eq x a) (.conj (.neg (phys .eq y a)) (phys .scattered b y))
      (iteF (phys .eq y a) (phys .scattered x b) (phys .scattered x y))
  | .onZero => phys .onZero x y
  | .eq => phys .eq x y

def separatePhysical (a : α) (r : Relation) (x y : α) : Formula α :=
  match r with
  | .on =>
    .conj (phys .on x y)
      (.disj (.neg (phys .on x a)) (phys .on y a))
  | .higher =>
    iteF (phys .eq x a) (.disj (phys .eq y a) (bottomHeight y))
      (iteF (phys .eq y a) .truth (phys .higher x y))
  | .scattered =>
    iteF (phys .eq x a) (.neg (phys .eq y a))
      (iteF (phys .eq y a) .truth (phys .scattered x y))
  | .onZero => phys .onZero x y
  | .eq => phys .eq x y

def putAtom (a b : α) (r : Relation) (x y : Object α) : Formula α :=
  match x, y with
  | some x, some y => putPhysical a b r x y
  | _, _ => .atom r x y

def separateAtom (a : α) (r : Relation) (x y : Object α) : Formula α :=
  match x, y with
  | some x, some y => separatePhysical a r x y
  | _, _ => .atom r x y

/-- Structural, simultaneous replacement: do not rewrite inside newly
inserted pre-state formulas. -/
def rewrite (replace : Relation → Object α → Object α → Formula α) :
    Formula α → Formula α
  | .atom r x y => replace r x y
  | .truth => .truth
  | .falsity => .falsity
  | .neg p => .neg (p.rewrite replace)
  | .conj p q => .conj (p.rewrite replace) (q.rewrite replace)
  | .disj p q => .disj (p.rewrite replace) (q.rewrite replace)
  | .all body => .all fun x => (body x).rewrite replace
  | .ex body => .ex fun x => (body x).rewrite replace

theorem rewrite_correct (replace : Relation → Object α → Object α → Formula α)
    (s s' frozen : Scene α)
    (atoms : ∀ r x y, (replace r x y).eval s frozen ↔ relation s' frozen r x y)
    (q : Formula α) :
    (q.rewrite replace).eval s frozen ↔ q.eval s' frozen := by
  induction q with
  | atom r x y => exact atoms r x y
  | truth => rfl
  | falsity => rfl
  | neg p ih => exact not_congr ih
  | conj p q hp hq => exact and_congr hp hq
  | disj p q hp hq => exact or_congr hp hq
  | all body ih => exact forall_congr' ih
  | ex body ih => exact exists_congr ih

section Updates
variable [DecidableEq α]

theorem putAtom_correct {s : Scene α} (wf : s.WellFormed)
    (frozen : Scene α) {a b : α} (ha : s.Clear a) (hb : s.Clear b)
    (r : Relation) (x y : Object α) :
    (putAtom a b r x y).eval s frozen ↔ relation (s.put a b) frozen r x y := by
  classical
  cases x with
  | none => cases y <;> cases r <;> rfl
  | some x =>
    cases y with
    | none => cases r <;> rfl
    | some y =>
      cases r with
      | on =>
        simpa only [putAtom, putPhysical, eval, phys, relation, Option.some.injEq]
          using (Scene.on_put ha hb x y).symm
      | onZero => rfl
      | eq => rfl
      | higher =>
        change (putPhysical a b .higher x y).eval s frozen ↔ (s.put a b).Higher x y
        rw [Scene.higher_put wf]
        simp only [putPhysical, iteF, eval, eval_nextHeight, phys, relation,
          Option.some.injEq]
        by_cases hx : x = a <;> by_cases hy : y = a <;>
          simp [Scene.higherPut, hx, hy]
      | scattered =>
        change (putPhysical a b .scattered x y).eval s frozen ↔
          (s.put a b).Scattered x y
        rw [Scene.scattered_put]
        simp only [putPhysical, iteF, eval, phys, relation, Option.some.injEq]
        by_cases hx : x = a <;> by_cases hy : y = a <;>
          simp [Scene.scatteredPut, hx, hy]

theorem separateAtom_correct {s : Scene α} (wf : s.WellFormed)
    (frozen : Scene α) {a : α} (ha : s.Clear a)
    (r : Relation) (x y : Object α) :
    (separateAtom a r x y).eval s frozen ↔ relation (s.separate a) frozen r x y := by
  classical
  cases x with
  | none => cases y <;> cases r <;> rfl
  | some x =>
    cases y with
    | none => cases r <;> rfl
    | some y =>
      cases r with
      | on => exact (Scene.on_separate wf ha x y).symm
      | onZero => rfl
      | eq => rfl
      | higher =>
        change (separatePhysical a .higher x y).eval s frozen ↔
          (s.separate a).Higher x y
        rw [Scene.higher_separate wf]
        simp only [separatePhysical, iteF, eval, eval_bottomHeight, phys,
          relation, Option.some.injEq]
        by_cases hx : x = a <;> by_cases hy : y = a <;>
          simp [Scene.higherSeparate, hx, hy]
      | scattered =>
        change (separatePhysical a .scattered x y).eval s frozen ↔
          (s.separate a).Scattered x y
        rw [Scene.scattered_separate wf ha]
        simp only [separatePhysical, iteF, eval, phys, relation, Option.some.injEq]
        by_cases hx : x = a <;> by_cases hy : y = a <;>
          simp [Scene.scatteredSeparate, hx, hy]

theorem rewrite_put {s : Scene α} (wf : s.WellFormed)
    (frozen : Scene α) {a b : α} (ha : s.Clear a) (hb : s.Clear b)
    (q : Formula α) :
    (q.rewrite (putAtom a b)).eval s frozen ↔ q.eval (s.put a b) frozen :=
  rewrite_correct _ _ _ _ (putAtom_correct wf frozen ha hb) q

theorem rewrite_separate {s : Scene α} (wf : s.WellFormed)
    (frozen : Scene α) {a : α} (ha : s.Clear a) (q : Formula α) :
    (q.rewrite (separateAtom a)).eval s frozen ↔ q.eval (s.separate a) frozen :=
  rewrite_correct _ _ _ _ (separateAtom_correct wf frozen ha) q

/-- Both operands must be clear BEFORE public Put. -/
def putGuard (a b : α) : Formula α :=
  .conj (clear a) (.conj (clear b) (.neg (phys .eq a b)))

def wpPut (a b : α) (q : Formula α) : Formula α :=
  .conj (putGuard a b) (q.rewrite (putAtom a b))

def wpSeparate (a : α) (q : Formula α) : Formula α :=
  .conj (clear a) (q.rewrite (separateAtom a))

/-- Legal operational semantics, independently of assertions and their rewrite. -/
def PutStep (s : Scene α) (a b : α) (s' : Scene α) : Prop :=
  s.Clear a ∧ s.Clear b ∧ a ≠ b ∧ s' = s.put a b

def SeparateStep (s : Scene α) (a : α) (s' : Scene α) : Prop :=
  s.Clear a ∧ s' = s.separate a

theorem wpPut_exact {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (a b : α) (q : Formula α) :
    (wpPut a b q).eval s frozen ↔
      ∃ s', PutStep s a b s' ∧ q.eval s' frozen := by
  simp only [wpPut, putGuard, eval, eval_clear, phys, relation, Option.some.injEq]
  constructor
  · rintro ⟨⟨ha, hb, hne⟩, hq⟩
    exact ⟨s.put a b, ⟨ha, hb, hne, rfl⟩, (rewrite_put wf frozen ha hb q).mp hq⟩
  · rintro ⟨s', ⟨ha, hb, hne, rfl⟩, hq⟩
    exact ⟨⟨ha, hb, hne⟩, (rewrite_put wf frozen ha hb q).mpr hq⟩

theorem wpSeparate_exact {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (a : α) (q : Formula α) :
    (wpSeparate a q).eval s frozen ↔
      ∃ s', SeparateStep s a s' ∧ q.eval s' frozen := by
  simp only [wpSeparate, eval, eval_clear]
  constructor
  · rintro ⟨ha, hq⟩
    exact ⟨s.separate a, ⟨ha, rfl⟩, (rewrite_separate wf frozen ha q).mp hq⟩
  · rintro ⟨s', ⟨ha, rfl⟩, hq⟩
    exact ⟨ha, (rewrite_separate wf frozen ha q).mpr hq⟩

/-- The original additive ON* rewrite, with the other predicate rewrites
unchanged. Its proof requires a singleton source. -/
def attachAtom (a b : α) (r : Relation) (x y : Object α) : Formula α :=
  match r with
  | .on => .disj (.atom .on x y)
      (.conj (.atom .on x (some a)) (.atom .on (some b) y))
  | _ => putAtom a b r x y

theorem attachAtom_correct {s : Scene α} (wf : s.WellFormed)
    (frozen : Scene α) {a b : α} (ha : s.Singleton a) (hb : s.Clear b)
    (r : Relation) (x y : Object α) :
    (attachAtom a b r x y).eval s frozen ↔ relation (s.put a b) frozen r x y := by
  cases r with
  | on =>
    cases x <;> cases y
    · simp [attachAtom, eval, relation]
    · simp [attachAtom, eval, relation]
    · simp [attachAtom, eval, relation]
    · exact (Scene.on_attach ha hb _ _).symm
  | higher | scattered | onZero | eq => exact putAtom_correct wf frozen ha.1 hb _ x y

theorem rewrite_attach {s : Scene α} (wf : s.WellFormed)
    (frozen : Scene α) {a b : α} (ha : s.Singleton a) (hb : s.Clear b)
    (q : Formula α) :
    (q.rewrite (attachAtom a b)).eval s frozen ↔ q.eval (s.put a b) frozen :=
  rewrite_correct _ _ _ _ (attachAtom_correct wf frozen ha hb) q

def singleton (a : α) : Formula α :=
  .conj (clear a) (allPhysical fun x => imp (phys .on a x) (phys .eq x a))

omit [DecidableEq α] in
@[simp] theorem eval_singleton (s frozen : Scene α) (a : α) :
    (singleton a).eval s frozen ↔ s.Singleton a := by
  simp only [singleton, eval, eval_clear, eval_allPhysical, eval_imp, phys,
    relation, Option.some.injEq, Scene.Singleton]

def wpAttach (a b : α) (q : Formula α) : Formula α :=
  .conj (.conj (singleton a) (.conj (clear b) (.neg (phys .eq a b))))
    (q.rewrite (attachAtom a b))

def AttachStep (s : Scene α) (a b : α) (s' : Scene α) : Prop :=
  s.Singleton a ∧ s.Clear b ∧ a ≠ b ∧ s' = s.put a b

theorem wpAttach_exact {s : Scene α} (wf : s.WellFormed) (frozen : Scene α)
    (a b : α) (q : Formula α) :
    (wpAttach a b q).eval s frozen ↔
      ∃ s', AttachStep s a b s' ∧ q.eval s' frozen := by
  simp only [wpAttach, eval, eval_singleton, eval_clear, phys, relation, Option.some.injEq]
  constructor
  · rintro ⟨⟨ha, hb, hne⟩, hq⟩
    exact ⟨s.put a b, ⟨ha, hb, hne, rfl⟩, (rewrite_attach wf frozen ha hb q).mp hq⟩
  · rintro ⟨s', ⟨ha, hb, hne, rfl⟩, hq⟩
    exact ⟨⟨ha, hb, hne⟩, (rewrite_attach wf frozen ha hb q).mpr hq⟩

/-- Under the agreed pre-state guard, the two-step logical decomposition
has exactly the same postcondition meaning as direct relocation. -/
theorem wpPut_decomposition {s : Scene α} (wf : s.WellFormed)
    (frozen : Scene α) {a b : α} (ha : s.Clear a) (hb : s.Clear b)
    (hne : a ≠ b) (q : Formula α) :
    (wpSeparate a (wpAttach a b q)).eval s frozen ↔ (wpPut a b q).eval s frozen := by
  have wfs := Scene.separate_wellFormed wf ha
  have has := Scene.separate_singleton wf ha
  have hbs := Scene.separate_clear wf ha hb
  simp only [wpSeparate, eval, eval_clear]
  rw [rewrite_separate wf frozen ha]
  rw [wpAttach_exact wfs, wpPut_exact wf]
  simp only [AttachStep, PutStep, ha, hb, has, hbs, true_and, Scene.separate_put s hne]

end Updates
end Formula
end RoboVerify
