import RoboVerify.FiniteDecision

set_option autoImplicit false

namespace RoboVerify.AFPreservation
open Fragments
variable {V D E A : Type}

theorem qf_embedding (q : QF (V ⊕ A))
    (M : Relation → D → D → Prop) (N : Relation → E → E → Prop) (f : E → D)
    (atoms : ∀ r x y, N r x y ↔ M r (f x) (f y))
    (ρ : V → E) (x : A → E) :
    q.eval N (Sum.elim ρ x) ↔ q.eval M (Sum.elim (f ∘ ρ) (f ∘ x)) := by
  have h := QF.eval_embedding q M N f atoms (Sum.elim ρ x)
  have he : f ∘ Sum.elim ρ x = Sum.elim (f ∘ ρ) (f ∘ x) := by
    funext v; cases v <;> rfl
  rw [he] at h; exact h

/-- Mutual embeddings preserving constants preserve CAV AF^R formulas.
The embeddings need not be inverse or surjective. -/
theorem af_mutual_embeddings (p : AF V)
    (M : Relation → D → D → Prop) (N : Relation → E → E → Prop)
    (f : E → D) (g : D → E)
    (hf : ∀ r x y, N r x y ↔ M r (f x) (f y))
    (hg : ∀ r x y, M r x y ↔ N r (g x) (g y))
    (ρ : V → D) (σ : V → E) (hρ : f ∘ σ = ρ) (hσ : g ∘ ρ = σ) :
    p.eval M ρ ↔ p.eval N σ := by
  induction p with
  | all b q =>
    constructor
    · intro h x
      apply (qf_embedding q M N f hf σ x).mpr
      rw [hρ]; exact h _
    · intro h x
      apply (qf_embedding q N M g hg ρ x).mpr
      rw [hσ]; exact h _
  | ex b q =>
    constructor
    · rintro ⟨x, hx⟩
      refine ⟨g ∘ x, ?_⟩
      have h := (qf_embedding q N M g hg ρ x).mp hx
      rw [hσ] at h; exact h
    · rintro ⟨x, hx⟩
      refine ⟨f ∘ x, ?_⟩
      have h := (qf_embedding q M N f hf σ x).mp hx
      rw [hρ] at h; exact h
  | conj p q hp hq => exact and_congr hp hq
  | disj p q hp hq => exact or_congr hp hq

/-- Two singleton columns and an unbounded complete column. The scene model
permits arbitrary object types, including this countable domain. -/
inductive Obj where
  | a | b | base | node : Nat → Obj
  deriving DecidableEq

def current : Scene Obj where
  root
    | .a => .a
    | .b => .b
    | .base | .node _ => .base
  level
    | .a | .b | .base => 0
    | .node n => n + 1

def colored (positive : Bool) (n : Nat) : Prop := n % 2 = if positive then 0 else 1

instance (positive : Bool) (n : Nat) : Decidable (colored positive n) :=
  inferInstanceAs (Decidable (n % 2 = if positive then 0 else 1))

/-- Colored nodes form a complete frozen column above base; uncolored nodes
are frozen singleton columns. Both frozen geometries are well formed. -/
def frozen (positive : Bool) : Scene Obj where
  root
    | .a => .a
    | .b => .b
    | .base => .base
    | .node n => if colored positive n then .base else .node n
  level
    | .a | .b | .base => 0
    | .node n => if colored positive n then n / 2 + 1 else 0

theorem current_wf : current.WellFormed := by
  constructor
  · intro x; cases x <;> rfl
  · intro x; cases x <;> rfl
  · intro x y hr hl
    cases x <;> cases y <;> simp_all [current]
  · intro x k hk
    cases x with
    | a => have he : k = 0 := Nat.eq_zero_of_le_zero hk; exact ⟨.a, rfl, he.symm⟩
    | b => have he : k = 0 := Nat.eq_zero_of_le_zero hk; exact ⟨.b, rfl, he.symm⟩
    | base => have he : k = 0 := Nat.eq_zero_of_le_zero hk; exact ⟨.base, rfl, he.symm⟩
    | node n =>
      by_cases he : k = 0
      · exact ⟨.base, rfl, he.symm⟩
      · exact ⟨.node (k - 1), rfl, by change k - 1 + 1 = k; omega⟩

theorem colored_shift (positive : Bool) (n : Nat) :
    colored positive n ↔ colored (!positive) (n + 1) := by
  cases positive <;> simp only [colored, Bool.not_false, Bool.not_true, Bool.false_eq_true,
    ite_true, ite_false] <;> omega

theorem colored_unique (positive : Bool) {n m : Nat}
    (hn : colored positive n) (hm : colored positive m) (h : n / 2 = m / 2) : n = m := by
  unfold colored at hn hm
  cases positive <;> simp_all <;> omega

theorem colored_order (positive : Bool) {n m : Nat}
    (hn : colored positive n) (hm : colored positive m) :
    m / 2 + 1 ≤ n / 2 + 1 ↔ m ≤ n := by
  unfold colored at hn hm
  cases positive <;> simp_all <;> omega


def shift : Obj → Obj
  | .a => .a
  | .b => .b
  | .base => .base
  | .node n => .node (n + 1)

theorem shift_inj (x y : Obj) : shift x = shift y ↔ x = y := by
  cases x <;> cases y <;> simp [shift]

theorem current_on_shift (x y : Obj) : current.On (shift x) (shift y) ↔ current.On x y := by
  cases x <;> cases y <;> simp [Scene.On, current, shift]

theorem current_higher_shift (x y : Obj) :
    current.Higher (shift x) (shift y) ↔ current.Higher x y := by
  cases x <;> cases y <;> simp [Scene.Higher, current, shift]

theorem current_scattered_shift (x y : Obj) :
    current.Scattered (shift x) (shift y) ↔ current.Scattered x y := by
  cases x <;> cases y <;> simp [Scene.Scattered, current, shift]

def color (positive : Bool) : Obj → Prop
  | .base => True
  | .node n => colored positive n
  | _ => False

theorem frozen_nodes (positive : Bool) (n m : Nat) :
    (frozen positive).On (.node n) (.node m) ↔
      n = m ∨ (colored positive n ∧ colored positive m ∧ m ≤ n) := by
  by_cases hn : colored positive n
  · by_cases hm : colored positive m
    · have ho := colored_order positive hn hm
      simp only [Scene.On, frozen, hn, hm, ite_true, true_and]
      rw [ho]; omega
    · have hne : n ≠ m := fun h => hm (h ▸ hn)
      simp [Scene.On, frozen, hn, hm, hne]
  · by_cases hm : colored positive m
    · have hne : n ≠ m := fun h => hn (h.symm ▸ hm)
      simp [Scene.On, frozen, hn, hm, hne]
    · simp [Scene.On, frozen, hn, hm]

theorem frozen_on (positive : Bool) (x y : Obj) :
    (frozen positive).On x y ↔
      x = y ∨ (color positive x ∧ color positive y ∧ current.On x y) := by
  cases x <;> cases y
  case node.node n m =>
    simpa [color, Scene.On, current] using frozen_nodes positive n m
  all_goals try (solve | simp [Scene.On, frozen, current, color])
  all_goals rename_i n
  all_goals by_cases hn : colored positive n <;> simp [Scene.On, frozen, current, color, hn]

theorem color_shift (positive : Bool) (x : Obj) :
    color (!positive) (shift x) ↔ color positive x := by
  cases x <;> simp only [color, shift]
  exact (colored_shift positive _).symm

theorem frozen_on_shift (positive : Bool) (x y : Obj) :
    (frozen (!positive)).On (shift x) (shift y) ↔ (frozen positive).On x y := by
  simp only [frozen_on, shift_inj, color_shift, current_on_shift]

def objectShift : Object Obj → Object Obj := Option.map shift

theorem atoms_shift (positive : Bool) (r : Relation) (x y : Object Obj) :
    relation current (frozen positive) r x y ↔
      relation current (frozen (!positive)) r (objectShift x) (objectShift y) := by
  cases r <;> cases x <;> cases y <;> simp [relation, objectShift, shift_inj,
    current_on_shift, current_higher_shift, current_scattered_shift, frozen_on_shift]

theorem frozen_wf (positive : Bool) : (frozen positive).WellFormed := by
  constructor
  · intro x
    cases x with
    | a => rfl
    | b => rfl
    | base => rfl
    | node n => by_cases hn : colored positive n <;> simp [frozen, hn]
  · intro x
    cases x with
    | a => rfl
    | b => rfl
    | base => rfl
    | node n => by_cases hn : colored positive n <;> simp [frozen, hn]
  · intro x y hr hl
    have hxy : (frozen positive).On x y := ⟨hr, Nat.le_of_eq hl.symm⟩
    have hyx : (frozen positive).On y x := ⟨hr.symm, Nat.le_of_eq hl⟩
    rcases (frozen_on positive x y).mp hxy with he | hx
    · exact he
    · rcases (frozen_on positive y x).mp hyx with he | hy
      · exact he.symm
      · exact Scene.on_antisymm current_wf hx.2.2 hy.2.2
  · intro x k hk
    cases x with
    | a => have he : k = 0 := Nat.eq_zero_of_le_zero hk; exact ⟨.a, rfl, he.symm⟩
    | b => have he : k = 0 := Nat.eq_zero_of_le_zero hk; exact ⟨.b, rfl, he.symm⟩
    | base => have he : k = 0 := Nat.eq_zero_of_le_zero hk; exact ⟨.base, rfl, he.symm⟩
    | node n =>
      by_cases hc : colored positive n
      · by_cases hz : k = 0
        · refine ⟨.base, ?_, hz.symm⟩; simp [frozen, hc]
        · let m := 2 * (k - 1) + if positive then 0 else 1
          have hm : colored positive m := by cases positive <;> simp [colored, m] <;> omega
          refine ⟨.node m, ?_, ?_⟩
          · simp [frozen, hc, hm]
          · simp only [frozen, hm, ite_true]
            dsimp [m]; cases positive <;> simp <;> omega
      · have he : k = 0 := by simp only [frozen, hc, ite_false] at hk; omega
        exact ⟨.node n, rfl, by simp [frozen, hc, he]⟩

inductive Name where
  | a | b | d
  deriving DecidableEq

def names : Option Name → Object Obj
  | none => none
  | some .a => some .a
  | some .b => some .b
  | some .d => some .base

theorem names_shift : objectShift ∘ names = names := by
  funext v
  cases v with
  | none => rfl
  | some v => cases v <;> rfl

def state (positive : Bool) : BlockProgram.State Obj Name := ⟨current, frozen positive, names⟩

theorem admissible (positive : Bool) : BlockProgram.Admissible (state positive) :=
  ⟨current_wf, frozen_wf positive, rfl⟩

theorem af_agree (p : AF (Option Name)) :
    p.eval (relation current (frozen false)) names ↔
      p.eval (relation current (frozen true)) names :=
  af_mutual_embeddings p _ _ objectShift objectShift (atoms_shift true) (atoms_shift false)
    names names names_shift names_shift

/-- An existential (hence CAV AF^R) postcondition:
there is x!=a at a's height with frozen ON*(x,d). -/
def post : AF (Option Name) := .ex .one
  (.conj (.neg (.atom .eq (.inr ()) (.inl (some .a))))
    (.conj (.atom .higher (.inl (some .a)) (.inr ()))
      (.conj (.atom .higher (.inr ()) (.inl (some .a)))
        (.atom .onZero (.inr ()) (.inl (some .d))))))

theorem post_put (positive : Bool) :
    post.eval (relation (current.put .a .b) (frozen positive)) names ↔ positive = true := by
  constructor
  · rintro ⟨x, hx⟩
    change x () ≠ some Obj.a ∧
      relation (current.put .a .b) (frozen positive) .higher (some .a) (x ()) ∧
      relation (current.put .a .b) (frozen positive) .higher (x ()) (some .a) ∧
      relation (current.put .a .b) (frozen positive) .onZero (x ()) (some .base) at hx
    cases ho : x () with
    | none => simp [ho, relation] at hx
    | some o =>
      cases o with
      | a => simp [ho] at hx
      | b => simp [ho, relation, Scene.Higher, Scene.put, current] at hx
      | base => simp [ho, relation, Scene.Higher, Scene.put, current] at hx
      | node n =>
        have hn : n = 0 := by
          have hh := hx.2.1
          simp only [ho, relation, Scene.Higher, Scene.put, current] at hh
          simp at hh
          omega
        subst n
        have hc := hx.2.2.2
        cases positive <;> simp [ho, relation, Scene.On, frozen, colored] at hc ⊢
  · intro hp
    subst positive
    refine ⟨fun _ => some (.node 0), ?_⟩
    simp [QF.eval, Sum.elim, names, relation, Scene.Higher, Scene.On, Scene.put,
      current, frozen, colored]

theorem clear_a : current.Clear .a := by
  intro x hx; cases x <;> simp_all [Scene.On, current]

theorem clear_b : current.Clear .b := by
  intro x hx; cases x <;> simp_all [Scene.On, current]

def wp : AE (Option Name) := Placement.wp (some .a) (some .b) none post.toAE

theorem wp_value (positive : Bool) :
    wp.eval (relation current (frozen positive)) names ↔ positive = true := by
  rw [wp, Placement.wp_exact current_wf (frozen positive) names
    (some .a) (some .b) none rfl]
  change (current.Clear .a ∧ current.Clear .b ∧ Obj.a ≠ Obj.b) ∧
    post.toAE.eval (relation (current.put .a .b) (frozen positive)) names ↔ _
  rw [AF.eval_toAE, post_put]
  simp [clear_a, clear_b]

/-- No CAV AF^R formula can express this exact Put WP over the stated
well-formed scene model. This is semantic nondefinability, not merely a
syntactic observation that one particular rewrite introduces alternation. -/
theorem no_af_wp : ¬∃ p : AF (Option Name),
    ∀ (α : Type) (s : BlockProgram.State α Name), BlockProgram.Admissible s →
      (p.eval (relation s.scene s.frozen) s.names ↔
        wp.eval (relation s.scene s.frozen) s.names) := by
  rintro ⟨p, hp⟩
  have htrue := (hp Obj (state true) (admissible true)).mpr ((wp_value true).mpr rfl)
  have hfalse := (af_agree p).mpr htrue
  have hw := (hp Obj (state false) (admissible false)).mp hfalse
  have impossible := (wp_value false).mp hw
  contradiction

/-- AF^R has both AE and EA representations. These helpers also let us use
the checked small-model theorem to obtain finite counterexamples below. -/
def afEA (p : AF V) : EA V := p.neg.toAE.refute

theorem eval_afEA [Nonempty D] (p : AF V) (M : Relation → D → D → Prop) (ρ : V → D) :
    (afEA p).eval M ρ ↔ p.eval M ρ := by
  classical
  simp [afEA, AE.eval_refute, AF.eval_toAE, AF.eval_neg]

def andEA (p q : EA V) : EA V := (p.negate.disj q.negate).refute

theorem eval_andEA [Nonempty D] (p q : EA V) (M : Relation → D → D → Prop) (ρ : V → D) :
    (andEA p q).eval M ρ ↔ p.eval M ρ ∧ q.eval M ρ := by
  classical
  simp [andEA, AE.eval_refute, AE.eval_disj, EA.eval_negate]

def exEA (b : Block) (p : EA (V ⊕ b.Index)) : EA V := (AE.all b p.negate).refute

theorem eval_exEA (b : Block) (p : EA (V ⊕ b.Index)) (M : Relation → D → D → Prop) (ρ : V → D) :
    (exEA b p).eval M ρ ↔ ∃ x, p.eval M (Sum.elim ρ x) := by
  classical
  simp [exEA, AE.eval_refute, AE.eval_all, EA.eval_negate, Classical.not_forall]

def legal : AF (Option Name) := Placement.legal (some .a) (some .b) none

def witness : AF (Option Name ⊕ Block.one.Index) :=
  (Placement.next (.inl (some .b)) (.inr ())).conj
    (AF.ofQF (.conj (.neg (.atom .eq (.inr ()) (.inl (some .a))))
      (.atom .onZero (.inr ()) (.inl (some .d)))))

/-- An EPR sufficient condition for this WP: a legal Put and a nonmoved
representative of the next height that lies in d's frozen column. -/
def sufficient : EA (Option Name) := andEA (afEA legal) (exEA .one (afEA witness))

theorem legal_state (positive : Bool) : legal.eval (relation current (frozen positive)) names :=
  (Placement.eval_legal current (frozen positive) names (some .a) (some .b) none
    .a .b rfl rfl rfl).mpr ⟨clear_a, clear_b, by decide⟩

theorem sufficient_true : sufficient.eval (relation current (frozen true)) names := by
  rw [sufficient, eval_andEA, eval_afEA, eval_exEA]
  refine ⟨legal_state true, fun _ => some (.node 0), ?_⟩
  rw [eval_afEA]
  change (Placement.next (.inl (some Name.b)) (.inr ())).eval
    (relation current (frozen true)) (Sum.elim names (fun _ => some (.node 0))) ∧ _
  constructor
  · apply (Placement.eval_next current (frozen true)
      (Sum.elim names (fun _ : Block.one.Index => some (Obj.node 0)))
      (.inl (some Name.b)) (.inr ())
      .b (.node 0) rfl rfl).mpr
    exact (HeightWitness.nextRep_iff current_wf .b (.node 0)).mpr rfl
  · simp [AF.eval_ofQF, QF.eval, Sum.elim, names, relation, Scene.On, frozen, colored]

theorem sufficient_sound {α : Type} (s : BlockProgram.State α Name)
    (hs : BlockProgram.Admissible s)
    (h : sufficient.eval (relation s.scene s.frozen) s.names) :
    wp.eval (relation s.scene s.frozen) s.names := by
  classical
  rw [sufficient, eval_andEA, eval_afEA, eval_exEA] at h
  obtain ⟨hl, x, hx⟩ := h
  rw [eval_afEA] at hx
  have hphys := hl.2.2
  simp only [AF.eval_ofQF, Placement.physical, QF.eval, relation, hs.2.2] at hphys
  obtain ⟨a, ha⟩ := Option.ne_none_iff_exists'.mp hphys.1
  obtain ⟨b, hb⟩ := Option.ne_none_iff_exists'.mp hphys.2.1
  have hl' := (Placement.eval_legal s.scene s.frozen s.names (some .a) (some .b) none
    a b ha hb hs.2.2).mp hl
  obtain ⟨u, hu⟩ := Placement.next_physical s.scene s.frozen (Sum.elim s.names x)
    (.inl (some .b)) (.inr ()) b hb hx.1
  have hn := (Placement.eval_next s.scene s.frozen (Sum.elim s.names x)
    (.inl (some .b)) (.inr ()) b u hb hu).mp hx.1
  have level := (HeightWitness.nextRep_iff hs.1 b u).mp hn
  have extra := hx.2
  simp only [AF.eval_ofQF, QF.eval, Sum.elim, relation] at extra
  have hu' : x () = some u := hu
  have ne : u ≠ a := by simpa only [hu', ha, Option.some.injEq] using extra.1
  rw [wp, Placement.wp_exact hs.1 s.frozen s.names (some .a) (some .b) none hs.2.2,
    ha, hb]
  refine ⟨hl', (AF.eval_toAE _ _ _).mpr ?_⟩
  refine ⟨fun _ => some u, ?_⟩
  change some u ≠ s.names (some .a) ∧
    relation (s.scene.put a b) s.frozen .higher (s.names (some .a)) (some u) ∧
    relation (s.scene.put a b) s.frozen .higher (some u) (s.names (some .a)) ∧
    relation (s.scene.put a b) s.frozen .onZero (some u) (s.names (some .d))
  refine ⟨by simpa [ha] using ne, ?_, ?_, ?_⟩
  · simp [ha, relation, Scene.Higher, Scene.put, ne, level]
  · simp [ha, relation, Scene.Higher, Scene.put, ne, level]
  · have hf := extra.2
    rw [hu'] at hf
    exact hf

/-- For EVERY proposed CAV AF^R rewrite there is a FINITE well-formed block
scene, with the operation enabled, where that rewrite disagrees with exact WP.
Thus the obstruction does not rely on admitting an infinite physical scene. -/
theorem finite_counterexample (p : AF (Option Name)) :
    ∃ n, ∃ s : BlockProgram.State (Fin n) Name,
      BlockProgram.Admissible s ∧ legal.eval (relation s.scene s.frozen) s.names ∧
      ¬(p.eval (relation s.scene s.frozen) s.names ↔
        wp.eval (relation s.scene s.frozen) s.names) := by
  classical
  let variables : List Name := [.a, .b, .d]
  have covers (v : Name) : v ∈ variables := by cases v <;> simp [variables]
  by_cases hp : p.eval (relation current (frozen false)) names
  · let query := andEA (afEA legal) (andEA (afEA p) wp.refute)
    have hq : query.eval (relation current (frozen false)) names := by
      simp only [query, eval_andEA, eval_afEA, AE.eval_refute]
      exact ⟨legal_state false, hp, fun h => Bool.noConfusion ((wp_value false).mp h)⟩
    obtain ⟨n, _, s, hs, _, _, h⟩ := BlockSmallModel.small_counterexample query variables covers
      (state false) (admissible false) hq
    simp only [query, eval_andEA, eval_afEA, AE.eval_refute] at h
    exact ⟨n, s, hs, h.1, fun he => h.2.2 (he.mp h.2.1)⟩
  · let query := andEA sufficient (afEA p.neg)
    have hq : query.eval (relation current (frozen true)) names := by
      simp only [query, eval_andEA, eval_afEA, AF.eval_neg]
      exact ⟨sufficient_true, fun h => hp ((af_agree p).mpr h)⟩
    obtain ⟨n, _, s, hs, _, _, h⟩ := BlockSmallModel.small_counterexample query variables covers
      (state true) (admissible true) hq
    simp only [query, eval_andEA, eval_afEA, AF.eval_neg] at h
    have hl := h.1
    rw [sufficient, eval_andEA, eval_afEA] at hl
    exact ⟨n, s, hs, hl.1, fun he => h.2 (he.mpr (sufficient_sound s hs h.1))⟩

def semanticPut {α : Type} [DecidableEq α] (s : BlockProgram.State α Name) : Prop :=
  let op : BlockProgram.Operation Name := .put (some .a) (some .b)
  (∃ t, op.Step s t) ∧ ∀ t, op.Step s t → post.eval (relation t.scene t.frozen) t.names

theorem wp_semanticPut_exact {α : Type} [DecidableEq α] (s : BlockProgram.State α Name)
    (hs : BlockProgram.Admissible s) :
    wp.eval (relation s.scene s.frozen) s.names ↔ semanticPut s := by
  have h := BlockProgram.Operation.wp_exact (.put (some Name.a) (some Name.b)) s hs post.toAE
  simpa only [wp, semanticPut, BlockProgram.Operation.wp, AF.eval_toAE] using h

/-- The failure applies to ANY exact WP rule in the existing AF^R language,
not just the particular implementation of Placement.wp. It holds even when
attention is restricted to finite scenes where the operation is enabled. -/
theorem no_exact_af_wp : ¬∃ p : AF (Option Name),
    ∀ n (s : BlockProgram.State (Fin n) Name), BlockProgram.Admissible s →
      legal.eval (relation s.scene s.frozen) s.names →
      (p.eval (relation s.scene s.frozen) s.names ↔ semanticPut s) := by
  rintro ⟨p, hp⟩
  obtain ⟨n, s, hs, hl, hne⟩ := finite_counterexample p
  exact hne ((hp n s hs hl).trans (wp_semanticPut_exact s hs).symm)

end RoboVerify.AFPreservation
