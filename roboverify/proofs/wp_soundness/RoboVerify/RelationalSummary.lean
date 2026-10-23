import RoboVerify.Formula

set_option autoImplicit false

namespace RoboVerify
namespace RelationalSummary

/-- These are state predicates. Frozen ON* and object equality are not havoced. -/
inductive MutableRelation where
  | on | higher | scattered
  deriving DecidableEq, Repr

abbrev State (α : Type) := MutableRelation → Object α → Object α → Prop
abbrev Frozen (α : Type) := Object α → Object α → Prop

inductive Kind where
  | higher | scattered
  deriving DecidableEq, Repr

variable {α : Type}

def target (kind : Kind) (s : State α) (left right : Object α) : Prop :=
  s (match kind with | .higher => .higher | .scattered => .scattered) left right

def Frame (s : State α) (moved : α) (next : State α) : Prop :=
  ∀ r x y, x ≠ some moved → y ≠ some moved → (next r x y ↔ s r x y)

/-- Six incident rows/columns may change. Every other atom retains its old value.
The candidate is a total relation valuation; its unused cells are irrelevant. -/
noncomputable def patch (s : State α) (moved : α) (candidate : State α) : State α := by
  classical
  exact fun r x y => if x = some moved ∨ y = some moved then candidate r x y else s r x y

theorem patch_frame (s candidate : State α) (moved : α) :
    Frame s moved (patch s moved candidate) := by
  intro r x y hx hy
  simp [patch, hx, hy]

/-- Every independently specified framed successor is represented by patch. -/
theorem patch_of_frame {s next : State α} {moved : α}
    (frame : Frame s moved next) : patch s moved next = next := by
  funext r x y
  apply propext
  by_cases h : x = some moved ∨ y = some moved
  · simp [patch, h]
  · have hx : x ≠ some moved := fun hx => h (Or.inl hx)
    have hy : y ≠ some moved := fun hy => h (Or.inr hy)
    simpa [patch, h] using (frame r x y hx hy).symm

def evalAtom (s : State α) (frozen : Frozen α) :
    Relation → Object α → Object α → Prop
  | .on, x, y => s .on x y
  | .higher, x, y => s .higher x y
  | .scattered, x, y => s .scattered x y
  | .onZero, x, y => frozen x y
  | .eq, x, y => x = y

def eval (s : State α) (frozen : Frozen α) : Formula α → Prop
  | .atom r x y => evalAtom s frozen r x y
  | .truth => True
  | .falsity => False
  | .neg p => ¬ eval s frozen p
  | .conj p q => eval s frozen p ∧ eval s frozen q
  | .disj p q => eval s frozen p ∨ eval s frozen q
  | .all body => ∀ x, eval s frozen (body x)
  | .ex body => ∃ x, eval s frozen (body x)

/-- The atomic WP rule, applied simultaneously to all mutable relations. -/
noncomputable def rewriteAtom (s candidate : State α) (frozen : Frozen α)
    (moved : α) (r : Relation) (x y : Object α) : Prop := by
  classical
  exact match r with
  | .onZero => frozen x y
  | .eq => x = y
  | .on => if x = some moved ∨ y = some moved then candidate .on x y else s .on x y
  | .higher => if x = some moved ∨ y = some moved then candidate .higher x y else s .higher x y
  | .scattered => if x = some moved ∨ y = some moved then candidate .scattered x y else s .scattered x y

/-- Interpretation of the structurally rewritten assertion, including quantifiers. -/
noncomputable def rewrite (s candidate : State α) (frozen : Frozen α) (moved : α) :
    Formula α → Prop
  | .atom r x y => rewriteAtom s candidate frozen moved r x y
  | .truth => True
  | .falsity => False
  | .neg p => ¬ rewrite s candidate frozen moved p
  | .conj p q => rewrite s candidate frozen moved p ∧ rewrite s candidate frozen moved q
  | .disj p q => rewrite s candidate frozen moved p ∨ rewrite s candidate frozen moved q
  | .all body => ∀ x, rewrite s candidate frozen moved (body x)
  | .ex body => ∃ x, rewrite s candidate frozen moved (body x)

theorem rewrite_correct (s candidate : State α) (frozen : Frozen α) (moved : α)
    (q : Formula α) :
    rewrite s candidate frozen moved q ↔ eval (patch s moved candidate) frozen q := by
  induction q with
  | atom r x y => cases r <;> rfl
  | truth => rfl
  | falsity => rfl
  | neg p ih => exact not_congr ih
  | conj p q hp hq => exact and_congr hp hq
  | disj p q hp hq => exact or_congr hp hq
  | all body ih => exact forall_congr' ih
  | ex body ih => exact exists_congr ih

/-- Execution is specified independently by a frame, admissibility, and endpoint
atom. Admissibility is explicit: no domain axiom is silently supplied by WP. -/
def Step (kind : Kind) (admissible : State α → Prop) (moved : α)
    (left right : Object α) (s next : State α) : Prop :=
  Frame s moved next ∧ admissible next ∧ target kind next left right

noncomputable def Candidate (kind : Kind) (admissible : State α → Prop) (moved : α)
    (left right : Object α) (s candidate : State α) : Prop :=
  admissible (patch s moved candidate) ∧ target kind (patch s moved candidate) left right

/-- Existence AND demonic correctness. An action without a successor faults. -/
noncomputable def wp (kind : Kind) (admissible : State α → Prop) (moved : α)
    (left right : Object α) (s : State α) (frozen : Frozen α) (q : Formula α) : Prop :=
  (∃ candidate, Candidate kind admissible moved left right s candidate) ∧
  ∀ candidate, Candidate kind admissible moved left right s candidate →
    rewrite s candidate frozen moved q

theorem wp_exact (kind : Kind) (admissible : State α → Prop) (moved : α)
    (left right : Object α) (s : State α) (frozen : Frozen α) (q : Formula α) :
    wp kind admissible moved left right s frozen q ↔
      (∃ next, Step kind admissible moved left right s next) ∧
      ∀ next, Step kind admissible moved left right s next → eval next frozen q := by
  constructor
  · rintro ⟨⟨candidate, hc⟩, hall⟩
    constructor
    · exact ⟨patch s moved candidate, patch_frame s candidate moved, hc⟩
    · intro next hn
      have heq := patch_of_frame hn.1
      have hc' : Candidate kind admissible moved left right s next := by
        simpa [Candidate, heq] using hn.2
      have hq := (rewrite_correct s next frozen moved q).mp (hall next hc')
      simpa [heq] using hq
  · rintro ⟨⟨next, hn⟩, hall⟩
    constructor
    · exact ⟨next, by simpa [Candidate, patch_of_frame hn.1] using hn.2⟩
    · intro candidate hc
      apply (rewrite_correct s candidate frozen moved q).mpr
      exact hall _ ⟨patch_frame s candidate moved, hc⟩

theorem wp_weakest (kind : Kind) (admissible : State α → Prop) (moved : α)
    (left right : Object α) (frozen : Frozen α) (q : Formula α)
    (pre : State α → Prop)
    (enabled : ∀ s, pre s → ∃ next, Step kind admissible moved left right s next)
    (correct : ∀ s next, pre s → Step kind admissible moved left right s next →
      eval next frozen q) :
    ∀ s, pre s → wp kind admissible moved left right s frozen q := by
  intro s hs
  exact (wp_exact kind admissible moved left right s frozen q).mpr
    ⟨enabled s hs, fun next hn => correct s next hs hn⟩

theorem no_successor_rejects (kind : Kind) (admissible : State α → Prop) (moved : α)
    (left right : Object α) (s : State α) (frozen : Frozen α) (q : Formula α)
    (none : ¬ ∃ next, Step kind admissible moved left right s next) :
    ¬ wp kind admissible moved left right s frozen q := by
  intro h
  exact none ((wp_exact kind admissible moved left right s frozen q).mp h).1

theorem one_bad_successor_rejects (kind : Kind) (admissible : State α → Prop)
    (moved : α) (left right : Object α) (s next : State α) (frozen : Frozen α)
    (q : Formula α) (step : Step kind admissible moved left right s next)
    (bad : ¬ eval next frozen q) :
    ¬ wp kind admissible moved left right s frozen q := by
  intro h
  exact bad (((wp_exact kind admissible moved left right s frozen q).mp h).2 next step)

/-- Refinement requires an actual endpoint relation, full frame, and admissibility.
They are premises to prove, not consequences of observing the endpoint atom. -/
theorem implementation_refines (implementation : State α → State α → Prop)
    (kind : Kind) (admissible : State α → Prop) (moved : α) (left right : Object α)
    (frame : ∀ s next, implementation s next → Frame s moved next)
    (domain : ∀ s next, implementation s next → admissible next)
    (endpoint : ∀ s next, implementation s next → target kind next left right)
    (s next : State α) (frozen : Frozen α) (q : Formula α)
    (pre : wp kind admissible moved left right s frozen q)
    (exec : implementation s next) : eval next frozen q := by
  exact ((wp_exact kind admissible moved left right s frozen q).mp pre).2 next
    ⟨frame s next exec, domain s next exec, endpoint s next exec⟩

/-- General demonic continuation rule, used to compose framed actions. -/
def must (step : State α → State α → Prop) (post : State α → Prop) (s : State α) : Prop :=
  (∃ next, step s next) ∧ ∀ next, step s next → post next

theorem sequential_sound (first second : State α → State α → Prop)
    (post : State α → Prop) (s middle last : State α)
    (pre : must first (must second post) s)
    (hfirst : first s middle) (hsecond : second middle last) : post last :=
  (pre.2 middle hfirst).2 last hsecond

theorem sequential_no_fault (first second : State α → State α → Prop)
    (post : State α → Prop) (s middle : State α)
    (pre : must first (must second post) s) (hfirst : first s middle) :
    ∃ last, second middle last := (pre.2 middle hfirst).1

/-- Two separated, same-height physical blocks: Higher is neither strict nor ON*. -/
def separatedPair : State Bool
  | .on, x, y => x = y
  | .higher, some _, some _ => True
  | .higher, none, none => True
  | .higher, _, _ => False
  | .scattered, some x, some y => x ≠ y
  | .scattered, _, _ => False

theorem higher_not_on :
    separatedPair .higher (some false) (some true) ∧
    ¬ separatedPair .on (some false) (some true) := by
  simp [separatedPair]

theorem higher_not_strict :
    separatedPair .higher (some false) (some true) ∧
    separatedPair .higher (some true) (some false) := ⟨True.intro, True.intro⟩

theorem scattered_allows_equal_height :
    separatedPair .scattered (some false) (some true) ∧
    separatedPair .higher (some false) (some true) ∧
    separatedPair .higher (some true) (some false) := by
  simp [separatedPair]

/-- Two separated columns with distinct heights; the table remains isolated. -/
def unequalPair : State Bool
  | .on, x, y => x = y
  | .higher, some x, some y => (if x then 1 else 0 : Nat) ≥ (if y then 1 else 0)
  | .higher, none, none => True
  | .higher, _, _ => False
  | .scattered, some x, some y => x ≠ y
  | .scattered, _, _ => False

theorem scattered_does_not_imply_higher :
    unequalPair .scattered (some false) (some true) ∧
    ¬ unequalPair .higher (some false) (some true) := by
  simp [unequalPair]

theorem scattered_does_not_imply_reverse_higher :
    unequalPair .scattered (some true) (some false) ∧
    ¬ unequalPair .higher (some false) (some true) := by
  simp [unequalPair]

/-- Blocks 0 and 1 share a column; block 2 has a different column. -/
def threeColumns : State Nat
  | .on, some x, some y => x / 2 = y / 2 ∧ x % 2 ≥ y % 2
  | .on, none, none => True
  | .on, _, _ => False
  | .higher, some x, some y => x % 2 ≥ y % 2
  | .higher, none, none => True
  | .higher, _, _ => False
  | .scattered, some x, some y => x / 2 ≠ y / 2
  | .scattered, _, _ => False

theorem scattered_does_not_imply_third_scattered :
    threeColumns .scattered (some 0) (some 2) ∧
    ¬ threeColumns .scattered (some 0) (some 1) := by
  simp [threeColumns]

/-- An irreflexive successor domain forbids Scattered(a,a); WP is not vacuous. -/
theorem scattered_self_rejects (s : State α) (frozen : Frozen α) (a : α) (q : Formula α) :
    ¬ wp .scattered (fun t => ∀ x, ¬ t .scattered x x) a (some a) (some a) s frozen q := by
  apply no_successor_rejects
  rintro ⟨next, _, domain, endpoint⟩
  exact domain (some a) endpoint

end RelationalSummary
end RoboVerify
