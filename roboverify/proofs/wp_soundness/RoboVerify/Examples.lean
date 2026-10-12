import RoboVerify.Program

set_option autoImplicit false

namespace RoboVerify.Examples

inductive Block where
  | a | b | d
  deriving DecidableEq, Repr
open Block

/-- A is initially on D; B is a separate singleton. -/
def initial : Scene Block where
  root | a => d | b => b | d => d
  level | a => 1 | b => 0 | d => 0

theorem initial_wellFormed : initial.WellFormed := by
  constructor
  · intro x; cases x <;> rfl
  · intro x; cases x <;> rfl
  · intro x y hr hl
    cases x <;> cases y <;> simp_all [initial]
  · intro x k hk
    cases x with
    | a =>
      have hk' : k = 0 ∨ k = 1 := by simp [initial] at hk; omega
      rcases hk' with rfl | rfl
      · exact ⟨d, rfl, rfl⟩
      · exact ⟨a, rfl, rfl⟩
    | b =>
      have hk' : k = 0 := by simpa [initial] using hk
      subst k
      exact ⟨b, rfl, rfl⟩
    | d =>
      have hk' : k = 0 := by simpa [initial] using hk
      subst k
      exact ⟨d, rfl, rfl⟩

theorem a_clear : initial.Clear a := by
  intro x h; cases x <;> simp_all [Scene.On, initial]

theorem b_clear : initial.Clear b := by
  intro x h; cases x <;> simp_all [Scene.On, initial]

/-- A concrete counterexample to using attachment's additive ON* rule for
general top-to-top relocation. Both operands meet the agreed clear guard. -/
theorem attachment_is_not_relocation :
    initial.WellFormed ∧ initial.Clear a ∧ initial.Clear b ∧ a ≠ b ∧
      (initial.On a d ∨ (initial.On a a ∧ initial.On b d)) ∧
      ¬(initial.put a b).On a d := by
  refine ⟨initial_wellFormed, a_clear, b_clear, by decide, ?_, ?_⟩
  · left; simp [Scene.On, initial]
  · simp [Scene.On, Scene.put, initial]

/-- The new guard rejects an occupied target, including A's old support D. -/
theorem occupied_target_rejected :
    ¬ (Formula.wpPut a d .truth).eval initial initial := by
  intro h
  have hd : initial.Clear d := (Formula.wpPut_exact initial_wellFormed initial
    a d .truth).mp h |>.choose_spec.1.2.1
  have impossible : a = d := hd a (by simp [Scene.On, initial])
  cases impossible

/-- General relocation preserves the structural premises on this legal state. -/
example : (initial.put a b).WellFormed :=
  Scene.put_wellFormed initial_wellFormed a_clear b_clear (by decide)

/-- Frozen entry reachability is retained while current reachability changes. -/
def relocatedPost : Formula Block :=
  .conj (.neg (Formula.phys .on a d)) (Formula.phys .onZero a d)

theorem relocation_wp_accepts :
    (Formula.wpPut a b relocatedPost).eval initial initial := by
  apply (Formula.wpPut_exact initial_wellFormed initial a b relocatedPost).mpr
  refine ⟨initial.put a b, ⟨a_clear, b_clear, by decide, rfl⟩, ?_⟩
  simp [relocatedPost, Formula.eval, Formula.phys, relation, Scene.On, Scene.put, initial]

/-- An example with alternation and negation, checked by the general theorem. -/
def quantifiedPost : Formula Block :=
  .all fun x => .ex fun y => .conj (.atom .eq x y)
    (.neg (.conj (.atom .on x y) (.neg (.atom .on x y))))

example :
    (Formula.wpPut a b quantifiedPost).eval initial initial := by
  apply (Formula.wpPut_exact initial_wellFormed initial a b quantifiedPost).mpr
  refine ⟨initial.put a b, ⟨a_clear, b_clear, by decide, rfl⟩, ?_⟩
  intro x
  exact ⟨x, rfl, fun h => h.2 h.1⟩

/-- These examples also use the actual program syntax and VC theorem. -/
def relocate : Command Block Unit :=
  .act (.put (.literal (some a)) (.literal (some b)))

def start : State Block Unit := ⟨initial, initial, fun _ => none⟩

theorem program_example :
    (¬ relocate.Fault start) ∧
      ∀ t, relocate.Exec start t → t.WellFormed ∧
        t.satisfies (fun _ => relocatedPost) := by
  exact Command.verified_program relocate (relocate.wp
      (fun _ => relocatedPost)) (fun _ => relocatedPost)
    True.intro
    (by intro s _ hp; exact hp) start initial_wellFormed relocation_wp_accepts

/-- Removing completeness permits a missing intermediate level and breaks the
quantified Higher rewrite. This witnesses the necessity of the model premise. -/
def gap : Scene Bool where
  root := fun _ => false
  level x := if x then 2 else 0

theorem gap_other_structure :
    (∀ x, gap.root (gap.root x) = gap.root x) ∧
    (∀ x, gap.level (gap.root x) = 0) ∧
    (∀ x y, gap.root x = gap.root y → gap.level x = gap.level y → x = y) := by
  refine ⟨?_, ?_, ?_⟩
  · intro x; rfl
  · intro x; rfl
  · intro x y _ hl
    cases x <;> cases y <;> simp_all [gap]

theorem missing_level_breaks_higher :
    gap.NextHeight false true ∧ ¬ gap.level true ≤ gap.level false + 1 := by
  constructor
  · right
    constructor
    · simp [Scene.Higher, gap]
    · intro t ht
      cases t <;> simp_all [Scene.Higher, gap]
  · decide

theorem gap_not_wellFormed : ¬ gap.WellFormed := by
  intro wf
  exact missing_level_breaks_higher.2
    ((Scene.nextHeight_iff wf false true).mp missing_level_breaks_higher.1)

end RoboVerify.Examples
