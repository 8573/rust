// HIR datatypes. See the [rustc guide] for more info.
//!
//! [rustc guide]: https://rust-lang.github.io/rustc-guide/hir.html

pub use self::BlockCheckMode::*;
pub use self::CaptureClause::*;
pub use self::FunctionRetTy::*;
pub use self::Mutability::*;
pub use self::PrimTy::*;
pub use self::UnOp::*;
pub use self::UnsafeSource::*;

use hir::def::Def;
use hir::def_id::{DefId, DefIndex, LocalDefId, CRATE_DEF_INDEX};
use util::nodemap::{NodeMap, FxHashSet};
use mir::mono::Linkage;

use syntax_pos::{Span, DUMMY_SP, symbol::InternedString};
use syntax::source_map::{self, Spanned};
use rustc_target::spec::abi::Abi;
use syntax::ast::{self, CrateSugar, Ident, Name, NodeId, DUMMY_NODE_ID, AsmDialect};
use syntax::ast::{Attribute, Lit, StrStyle, FloatTy, IntTy, UintTy};
use syntax::attr::InlineAttr;
use syntax::ext::hygiene::SyntaxContext;
use syntax::symbol::{Symbol, keywords};
use syntax::tokenstream::TokenStream;
use syntax::util::parser::ExprPrecedence;
use ty::AdtKind;
use ty::query::Providers;

use rustc_data_structures::sync::{ParallelIterator, par_iter, Send, Sync, scope};

use arena::{TypedArena, SyncDroplessArena};
use serialize::{self, Encoder, Encodable, Decoder, Decodable};
use std::collections::BTreeMap;
use std::fmt;

use self::ptr::P;

/// HIR doesn't commit to a concrete storage type and has its own alias for a vector.
/// It can be `Vec`, `P<'a, [T]>` or potentially `Box<[T]>`, or some other container with similar
/// behavior. Unlike AST, HIR is mostly a static structure, so we can use an owned slice instead
/// of `Vec` to avoid keeping extra capacity.
pub type HirVec<'a, T> = P<'a, [T]>;
pub type ThinHirVec<'a, T> = P<'a, P<'a, [T]>>;

pub mod ptr;
pub mod check_attr;
pub mod def;
pub mod def_id;
pub mod intravisit;
pub mod itemlikevisit;
pub mod lowering;
pub mod map;
pub mod pat_util;
pub mod print;

#[derive(Default)]
pub struct Arenas {
    token_streams: TypedArena<TokenStream>,
    attrs: TypedArena<Attribute>,
    inline_asms: TypedArena<InlineAsm>,
    lits: TypedArena<Lit>,
}

/// A HirId uniquely identifies a node in the HIR of the current crate. It is
/// composed of the `owner`, which is the DefIndex of the directly enclosing
/// hir::Item, hir::TraitItem, or hir::ImplItem (i.e., the closest "item-like"),
/// and the `local_id` which is unique within the given owner.
///
/// This two-level structure makes for more stable values: One can move an item
/// around within the source code, or add or remove stuff before it, without
/// the local_id part of the HirId changing, which is a very useful property in
/// incremental compilation where we have to persist things through changes to
/// the code base.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct HirId {
    pub owner: DefIndex,
    pub local_id: ItemLocalId,
}

impl HirId {
    pub fn owner_def_id(self) -> DefId {
        DefId::local(self.owner)
    }

    pub fn owner_local_def_id(self) -> LocalDefId {
        LocalDefId::from_def_id(DefId::local(self.owner))
    }
}

impl serialize::UseSpecializedEncodable for HirId {
    fn default_encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        let HirId {
            owner,
            local_id,
        } = *self;

        owner.encode(s)?;
        local_id.encode(s)
    }
}

impl serialize::UseSpecializedDecodable for HirId {
    fn default_decode<D: Decoder>(d: &mut D) -> Result<HirId, D::Error> {
        let owner = DefIndex::decode(d)?;
        let local_id = ItemLocalId::decode(d)?;

        Ok(HirId {
            owner,
            local_id
        })
    }
}

// hack to ensure that we don't try to access the private parts of `ItemLocalId` in this module
mod item_local_id_inner {
    use rustc_data_structures::indexed_vec::Idx;
    /// An `ItemLocalId` uniquely identifies something within a given "item-like",
    /// that is within a hir::Item, hir::TraitItem, or hir::ImplItem. There is no
    /// guarantee that the numerical value of a given `ItemLocalId` corresponds to
    /// the node's position within the owning item in any way, but there is a
    /// guarantee that the `LocalItemId`s within an owner occupy a dense range of
    /// integers starting at zero, so a mapping that maps all or most nodes within
    /// an "item-like" to something else can be implement by a `Vec` instead of a
    /// tree or hash map.
    newtype_index! {
        pub struct ItemLocalId { .. }
    }
}

pub use self::item_local_id_inner::ItemLocalId;

/// The `HirId` corresponding to CRATE_NODE_ID and CRATE_DEF_INDEX
pub const CRATE_HIR_ID: HirId = HirId {
    owner: CRATE_DEF_INDEX,
    local_id: ItemLocalId::from_u32_const(0)
};

pub const DUMMY_HIR_ID: HirId = HirId {
    owner: CRATE_DEF_INDEX,
    local_id: DUMMY_ITEM_LOCAL_ID,
};

pub const DUMMY_ITEM_LOCAL_ID: ItemLocalId = ItemLocalId::MAX;

#[derive(Clone, RustcEncodable, RustcDecodable, Copy, HashStable)]
pub struct Label {
    pub ident: Ident,
}

impl fmt::Debug for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "label({:?})", self.ident)
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Copy, HashStable)]
pub struct Lifetime {
    pub id: NodeId,
    pub span: Span,

    /// Either "'a", referring to a named lifetime definition,
    /// or "" (aka keywords::Invalid), for elision placeholders.
    ///
    /// HIR lowering inserts these placeholders in type paths that
    /// refer to type definitions needing lifetime parameters,
    /// `&T` and `&mut T`, and trait objects without `... + 'a`.
    pub name: LifetimeName,
}

#[derive(Debug, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy, HashStable)]
pub enum ParamName {
    /// Some user-given name like `T` or `'x`.
    Plain(Ident),

    /// Synthetic name generated when user elided a lifetime in an impl header,
    /// e.g., the lifetimes in cases like these:
    ///
    ///     impl Foo for &u32
    ///     impl Foo<'_> for u32
    ///
    /// in that case, we rewrite to
    ///
    ///     impl<'f> Foo for &'f u32
    ///     impl<'f> Foo<'f> for u32
    ///
    /// where `'f` is something like `Fresh(0)`. The indices are
    /// unique per impl, but not necessarily continuous.
    Fresh(usize),

    /// Indicates an illegal name was given and an error has been
    /// repored (so we should squelch other derived errors). Occurs
    /// when e.g., `'_` is used in the wrong place.
    Error,
}

impl ParamName {
    pub fn ident(&self) -> Ident {
        match *self {
            ParamName::Plain(ident) => ident,
            ParamName::Error | ParamName::Fresh(_) => keywords::UnderscoreLifetime.ident(),
        }
    }

    pub fn modern(&self) -> ParamName {
        match *self {
            ParamName::Plain(ident) => ParamName::Plain(ident.modern()),
            param_name => param_name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy, HashStable)]
pub enum LifetimeName {
    /// User-given names or fresh (synthetic) names.
    Param(ParamName),

    /// User typed nothing. e.g., the lifetime in `&u32`.
    Implicit,

    /// Indicates an error during lowering (usually `'_` in wrong place)
    /// that was already reported.
    Error,

    /// User typed `'_`.
    Underscore,

    /// User wrote `'static`
    Static,
}

impl LifetimeName {
    pub fn ident(&self) -> Ident {
        match *self {
            LifetimeName::Implicit => keywords::Invalid.ident(),
            LifetimeName::Error => keywords::Invalid.ident(),
            LifetimeName::Underscore => keywords::UnderscoreLifetime.ident(),
            LifetimeName::Static => keywords::StaticLifetime.ident(),
            LifetimeName::Param(param_name) => param_name.ident(),
        }
    }

    pub fn is_elided(&self) -> bool {
        match self {
            LifetimeName::Implicit | LifetimeName::Underscore => true,

            // It might seem surprising that `Fresh(_)` counts as
            // *not* elided -- but this is because, as far as the code
            // in the compiler is concerned -- `Fresh(_)` variants act
            // equivalently to "some fresh name". They correspond to
            // early-bound regions on an impl, in other words.
            LifetimeName::Error | LifetimeName::Param(_) | LifetimeName::Static => false,
        }
    }

    fn is_static(&self) -> bool {
        self == &LifetimeName::Static
    }

    pub fn modern(&self) -> LifetimeName {
        match *self {
            LifetimeName::Param(param_name) => LifetimeName::Param(param_name.modern()),
            lifetime_name => lifetime_name,
        }
    }
}

impl fmt::Display for Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.name.ident().fmt(f)
    }
}

impl fmt::Debug for Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f,
               "lifetime({}: {})",
               self.id,
               print::to_string(print::NO_ANN, |s| s.print_lifetime(self)))
    }
}

impl Lifetime {
    pub fn is_elided(&self) -> bool {
        self.name.is_elided()
    }

    pub fn is_static(&self) -> bool {
        self.name.is_static()
    }
}

/// A "Path" is essentially Rust's notion of a name; for instance:
/// `std::cmp::PartialEq`. It's represented as a sequence of identifiers,
/// along with a bunch of supporting information.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, HashStable)]
pub struct Path<'a> {
    pub span: Span,
    /// The definition that the path resolved to.
    pub def: Def,
    /// The segments in the path: the things separated by `::`.
    pub segments: HirVec<'a, PathSegment<'a>>,
}

impl Path<'_> {
    pub fn is_global(&self) -> bool {
        !self.segments.is_empty() && self.segments[0].ident.name == keywords::PathRoot.name()
    }
}

impl fmt::Debug for Path<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "path({})", self)
    }
}

impl fmt::Display for Path<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", print::to_string(print::NO_ANN, |s| s.print_path(self, false)))
    }
}

/// A segment of a path: an identifier, an optional lifetime, and a set of
/// types.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct PathSegment<'a> {
    /// The identifier portion of this path segment.
    #[stable_hasher(project(name))]
    pub ident: Ident,
    // `id` and `def` are optional. We currently only use these in save-analysis,
    // any path segments without these will not have save-analysis info and
    // therefore will not have 'jump to def' in IDEs, but otherwise will not be
    // affected. (In general, we don't bother to get the defs for synthesized
    // segments, only for segments which have come from the AST).
    pub id: Option<NodeId>,
    pub def: Option<Def>,

    /// Type/lifetime parameters attached to this path. They come in
    /// two flavors: `Path<A,B,C>` and `Path(A,B) -> C`. Note that
    /// this is more than just simple syntactic sugar; the use of
    /// parens affects the region binding rules, so we preserve the
    /// distinction.
    pub args: Option<P<'a, GenericArgs<'a>>>,

    /// Whether to infer remaining type parameters, if any.
    /// This only applies to expression and pattern paths, and
    /// out of those only the segments with no type parameters
    /// to begin with, e.g., `Vec::new` is `<Vec<..>>::new::<..>`.
    pub infer_types: bool,
}

impl<'a> PathSegment<'a> {
    /// Convert an identifier to the corresponding segment.
    pub fn from_ident(ident: Ident) -> Self {
        PathSegment {
            ident,
            id: None,
            def: None,
            infer_types: true,
            args: None,
        }
    }

    pub fn new(
        arena: &'a SyncDroplessArena,
        ident: Ident,
        id: Option<NodeId>,
        def: Option<Def>,
        args: GenericArgs<'a>,
        infer_types: bool,
    ) -> Self {
        PathSegment {
            ident,
            id,
            def,
            infer_types,
            args: if args.is_empty() {
                None
            } else {
                Some(P::alloc(arena, args))
            }
        }
    }

    // FIXME: hack required because you can't create a static
    // `GenericArgs`, so you can't just return a `&GenericArgs`.
    pub fn with_generic_args<F, R>(&self, f: F) -> R
        where F: FnOnce(&GenericArgs<'a>) -> R
    {
        let dummy = GenericArgs::none();
        f(if let Some(ref args) = self.args {
            &args
        } else {
            &dummy
        })
    }
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum GenericArg<'a> {
    Lifetime(Lifetime),
    Type(Ty<'a>),
}

impl GenericArg<'_> {
    pub fn span(&self) -> Span {
        match self {
            GenericArg::Lifetime(l) => l.span,
            GenericArg::Type(t) => t.span,
        }
    }

    pub fn id(&self) -> NodeId {
        match self {
            GenericArg::Lifetime(l) => l.id,
            GenericArg::Type(t) => t.id,
        }
    }
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct GenericArgs<'a> {
    /// The generic arguments for this path segment.
    pub args: HirVec<'a, GenericArg<'a>>,
    /// Bindings (equality constraints) on associated types, if present.
    /// E.g., `Foo<A = Bar>`.
    pub bindings: HirVec<'a, TypeBinding<'a>>,
    /// Were arguments written in parenthesized form `Fn(T) -> U`?
    /// This is required mostly for pretty-printing and diagnostics,
    /// but also for changing lifetime elision rules to be "function-like".
    pub parenthesized: bool,
}

impl<'a> GenericArgs<'a> {
    pub fn none() -> Self {
        Self {
            args: HirVec::new(),
            bindings: HirVec::new(),
            parenthesized: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.args.is_empty() && self.bindings.is_empty() && !self.parenthesized
    }

    pub fn inputs(&self) -> &[Ty<'a>] {
        if self.parenthesized {
            for arg in &self.args {
                match arg {
                    GenericArg::Lifetime(_) => {}
                    GenericArg::Type(ref ty) => {
                        if let TyKind::Tup(ref tys) = ty.node {
                            return tys;
                        }
                        break;
                    }
                }
            }
        }
        bug!("GenericArgs::inputs: not a `Fn(T) -> U`");
    }

    pub fn own_counts(&self) -> GenericParamCount {
        // We could cache this as a property of `GenericParamCount`, but
        // the aim is to refactor this away entirely eventually and the
        // presence of this method will be a constant reminder.
        let mut own_counts: GenericParamCount = Default::default();

        for arg in &self.args {
            match arg {
                GenericArg::Lifetime(_) => own_counts.lifetimes += 1,
                GenericArg::Type(_) => own_counts.types += 1,
            };
        }

        own_counts
    }
}

/// A modifier on a bound, currently this is only used for `?Sized`, where the
/// modifier is `Maybe`. Negative bounds should also be handled here.
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, HashStable)]
pub enum TraitBoundModifier {
    None,
    Maybe,
}

/// The AST represents all type param bounds as types.
/// `typeck::collect::compute_bounds` matches these against
/// the "special" built-in traits (see `middle::lang_items`) and
/// detects `Copy`, `Send` and `Sync`.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum GenericBound<'a> {
    Trait(PolyTraitRef<'a>, TraitBoundModifier),
    Outlives(Lifetime),
}

impl GenericBound<'_> {
    pub fn span(&self) -> Span {
        match self {
            &GenericBound::Trait(ref t, ..) => t.span,
            &GenericBound::Outlives(ref l) => l.span,
        }
    }
}

pub type GenericBounds<'a> = HirVec<'a, GenericBound<'a>>;

#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum LifetimeParamKind {
    // Indicates that the lifetime definition was explicitly declared (e.g., in
    // `fn foo<'a>(x: &'a u8) -> &'a u8 { x }`).
    Explicit,

    // Indicates that the lifetime definition was synthetically added
    // as a result of an in-band lifetime usage (e.g., in
    // `fn foo(x: &'a u8) -> &'a u8 { x }`).
    InBand,

    // Indication that the lifetime was elided (e.g., in both cases in
    // `fn foo(x: &u8) -> &'_ u8 { x }`).
    Elided,

    // Indication that the lifetime name was somehow in error.
    Error,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum GenericParamKind<'a> {
    /// A lifetime definition (e.g., `'a: 'b + 'c + 'd`).
    Lifetime {
        kind: LifetimeParamKind,
    },
    Type {
        default: Option<P<'a, Ty<'a>>>,
        synthetic: Option<SyntheticTyParamKind>,
    }
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct GenericParam<'a> {
    pub id: NodeId,
    pub name: ParamName,
    pub attrs: HirVec<'a, Attribute>,
    pub bounds: GenericBounds<'a>,
    pub span: Span,
    pub pure_wrt_drop: bool,

    pub kind: GenericParamKind<'a>,
}

#[derive(Default)]
pub struct GenericParamCount {
    pub lifetimes: usize,
    pub types: usize,
}

/// Represents lifetimes and type parameters attached to a declaration
/// of a function, enum, trait, etc.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct Generics<'a> {
    pub params: HirVec<'a, GenericParam<'a>>,
    pub where_clause: WhereClause<'a>,
    pub span: Span,
}

impl Generics<'_> {
    pub fn empty<'a>() -> Generics<'a> {
        Generics {
            params: HirVec::new(),
            where_clause: WhereClause {
                id: DUMMY_NODE_ID,
                predicates: HirVec::new(),
            },
            span: DUMMY_SP,
        }
    }

    pub fn own_counts(&self) -> GenericParamCount {
        // We could cache this as a property of `GenericParamCount`, but
        // the aim is to refactor this away entirely eventually and the
        // presence of this method will be a constant reminder.
        let mut own_counts: GenericParamCount = Default::default();

        for param in &self.params {
            match param.kind {
                GenericParamKind::Lifetime { .. } => own_counts.lifetimes += 1,
                GenericParamKind::Type { .. } => own_counts.types += 1,
            };
        }

        own_counts
    }

    pub fn get_named(&self, name: &InternedString) -> Option<&GenericParam> {
        for param in &self.params {
            if *name == param.name.ident().as_interned_str() {
                return Some(param);
            }
        }
        None
    }
}

/// Synthetic Type Parameters are converted to an other form during lowering, this allows
/// to track the original form they had. Useful for error messages.
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, HashStable)]
pub enum SyntheticTyParamKind {
    ImplTrait
}

/// A `where` clause in a definition
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct WhereClause<'a> {
    pub id: NodeId,
    pub predicates: HirVec<'a, WherePredicate<'a>>,
}

impl WhereClause<'_> {
    pub fn span(&self) -> Option<Span> {
        self.predicates.iter().map(|predicate| predicate.span())
            .fold(None, |acc, i| match (acc, i) {
                (None, i) => Some(i),
                (Some(acc), i) => {
                    Some(acc.to(i))
                }
            })
    }
}

/// A single predicate in a `where` clause
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum WherePredicate<'a> {
    /// A type binding (e.g., `for<'c> Foo: Send + Clone + 'c`).
    BoundPredicate(WhereBoundPredicate<'a>),
    /// A lifetime predicate (e.g., `'a: 'b + 'c`).
    RegionPredicate(WhereRegionPredicate<'a>),
    /// An equality predicate (unsupported).
    EqPredicate(WhereEqPredicate<'a>),
}

impl WherePredicate<'_> {
    pub fn span(&self) -> Span {
        match self {
            &WherePredicate::BoundPredicate(ref p) => p.span,
            &WherePredicate::RegionPredicate(ref p) => p.span,
            &WherePredicate::EqPredicate(ref p) => p.span,
        }
    }
}

/// A type bound, eg `for<'c> Foo: Send+Clone+'c`
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct WhereBoundPredicate<'a> {
    pub span: Span,
    /// Any generics from a `for` binding
    pub bound_generic_params: HirVec<'a, GenericParam<'a>>,
    /// The type being bounded
    pub bounded_ty: P<'a, Ty<'a>>,
    /// Trait and lifetime bounds (`Clone+Send+'static`)
    pub bounds: GenericBounds<'a>,
}

/// A lifetime predicate, e.g., `'a: 'b+'c`
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct WhereRegionPredicate<'a> {
    pub span: Span,
    pub lifetime: Lifetime,
    pub bounds: GenericBounds<'a>,
}

/// An equality predicate (unsupported), e.g., `T=int`
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct WhereEqPredicate<'a> {
    pub id: NodeId,
    pub span: Span,
    pub lhs_ty: P<'a, Ty<'a>>,
    pub rhs_ty: P<'a, Ty<'a>>,
}

/// The top-level data structure that stores the entire contents of
/// the crate currently being compiled.
///
/// For more details, see the [rustc guide].
///
/// [rustc guide]: https://rust-lang.github.io/rustc-guide/hir.html
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Crate<'a> {
    pub module: Mod<'a>,
    pub attrs: HirVec<'a, Attribute>,
    pub span: Span,
    pub exported_macros: HirVec<'a, MacroDef<'a>>,

    // N.B., we use a BTreeMap here so that `visit_all_items` iterates
    // over the ids in increasing order. In principle it should not
    // matter what order we visit things in, but in *practice* it
    // does, because it can affect the order in which errors are
    // detected, which in turn can make compile-fail tests yield
    // slightly different results.
    pub items: BTreeMap<NodeId, Item<'a>>,

    pub trait_items: BTreeMap<TraitItemId, TraitItem<'a>>,
    pub impl_items: BTreeMap<ImplItemId, ImplItem<'a>>,
    pub bodies: BTreeMap<BodyId, Body<'a>>,
    pub trait_impls: BTreeMap<DefId, Vec<NodeId>>,
    pub trait_auto_impl: BTreeMap<DefId, NodeId>,

    /// A list of the body ids written out in the order in which they
    /// appear in the crate. If you're going to process all the bodies
    /// in the crate, you should iterate over this list rather than the keys
    /// of bodies.
    pub body_ids: Vec<BodyId>,
}

impl Crate<'_> {
    pub fn item(&self, id: NodeId) -> &Item {
        &self.items[&id]
    }

    pub fn trait_item(&self, id: TraitItemId) -> &TraitItem {
        &self.trait_items[&id]
    }

    pub fn impl_item(&self, id: ImplItemId) -> &ImplItem {
        &self.impl_items[&id]
    }

    /// Visits all items in the crate in some deterministic (but
    /// unspecified) order. If you just need to process every item,
    /// but don't care about nesting, this method is the best choice.
    ///
    /// If you do care about nesting -- usually because your algorithm
    /// follows lexical scoping rules -- then you want a different
    /// approach. You should override `visit_nested_item` in your
    /// visitor and then call `intravisit::walk_crate` instead.
    pub fn visit_all_item_likes<'hir, V>(&'hir self, visitor: &mut V)
        where V: itemlikevisit::ItemLikeVisitor<'hir>
    {
        for (_, item) in &self.items {
            visitor.visit_item(item);
        }

        for (_, trait_item) in &self.trait_items {
            visitor.visit_trait_item(trait_item);
        }

        for (_, impl_item) in &self.impl_items {
            visitor.visit_impl_item(impl_item);
        }
    }

    /// A parallel version of visit_all_item_likes
    pub fn par_visit_all_item_likes<'hir, V>(&'hir self, visitor: &V)
        where V: itemlikevisit::ParItemLikeVisitor<'hir> + Sync + Send
    {
        scope(|s| {
            s.spawn(|_| {
                par_iter(&self.items).for_each(|(_, item)| {
                    visitor.visit_item(item);
                });
            });

            s.spawn(|_| {
                par_iter(&self.trait_items).for_each(|(_, trait_item)| {
                    visitor.visit_trait_item(trait_item);
                });
            });

            s.spawn(|_| {
                par_iter(&self.impl_items).for_each(|(_, impl_item)| {
                    visitor.visit_impl_item(impl_item);
                });
            });
        });
    }

    pub fn body(&self, id: BodyId) -> &Body {
        &self.bodies[&id]
    }
}

/// A macro definition, in this crate or imported from another.
///
/// Not parsed directly, but created on macro import or `macro_rules!` expansion.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct MacroDef<'a> {
    pub name: Name,
    pub vis: Visibility<'a>,
    pub attrs: HirVec<'a, Attribute>,
    pub id: NodeId,
    pub span: Span,
    pub body: P<'a, TokenStream>,
    pub legacy: bool,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct Block<'a> {
    /// Statements in a block
    pub stmts: HirVec<'a, Stmt<'a>>,
    /// An expression at the end of the block
    /// without a semicolon, if any
    pub expr: Option<P<'a, Expr<'a>>>,
    #[stable_hasher(ignore)]
    pub id: NodeId,
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    /// Distinguishes between `unsafe { ... }` and `{ ... }`
    pub rules: BlockCheckMode,
    pub span: Span,
    /// If true, then there may exist `break 'a` values that aim to
    /// break out of this block early.
    /// Used by `'label: {}` blocks and by `catch` statements.
    pub targeted_by_break: bool,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, HashStable)]
pub struct Pat<'a> {
    #[stable_hasher(ignore)]
    pub id: NodeId,
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub node: PatKind<'a>,
    pub span: Span,
}

impl fmt::Debug for Pat<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pat({}: {})", self.id,
               print::to_string(print::NO_ANN, |s| s.print_pat(self)))
    }
}

impl Pat<'_> {
    // FIXME(#19596) this is a workaround, but there should be a better way
    fn walk_<G>(&self, it: &mut G) -> bool
        where G: FnMut(&Pat) -> bool
    {
        if !it(self) {
            return false;
        }

        match self.node {
            PatKind::Binding(.., Some(ref p)) => p.walk_(it),
            PatKind::Struct(_, ref fields, _) => {
                fields.iter().all(|field| field.node.pat.walk_(it))
            }
            PatKind::TupleStruct(_, ref s, _) | PatKind::Tuple(ref s, _) => {
                s.iter().all(|p| p.walk_(it))
            }
            PatKind::Box(ref s) | PatKind::Ref(ref s, _) => {
                s.walk_(it)
            }
            PatKind::Slice(ref before, ref slice, ref after) => {
                before.iter()
                      .chain(slice.iter())
                      .chain(after.iter())
                      .all(|p| p.walk_(it))
            }
            PatKind::Wild |
            PatKind::Lit(_) |
            PatKind::Range(..) |
            PatKind::Binding(..) |
            PatKind::Path(_) => {
                true
            }
        }
    }

    pub fn walk<F>(&self, mut it: F) -> bool
        where F: FnMut(&Pat) -> bool
    {
        self.walk_(&mut it)
    }
}

/// A single field in a struct pattern
///
/// Patterns like the fields of Foo `{ x, ref y, ref mut z }`
/// are treated the same as` x: x, y: ref y, z: ref mut z`,
/// except is_shorthand is true
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct FieldPat<'a> {
    #[stable_hasher(ignore)]
    pub id: NodeId,
    /// The identifier for the field
    #[stable_hasher(project(name))]
    pub ident: Ident,
    /// The pattern the field is destructured to
    pub pat: P<'a, Pat<'a>>,
    pub is_shorthand: bool,
}

/// Explicit binding annotations given in the HIR for a binding. Note
/// that this is not the final binding *mode* that we infer after type
/// inference.
#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, Copy, HashStable)]
pub enum BindingAnnotation {
    /// No binding annotation given: this means that the final binding mode
    /// will depend on whether we have skipped through a `&` reference
    /// when matching. For example, the `x` in `Some(x)` will have binding
    /// mode `None`; if you do `let Some(x) = &Some(22)`, it will
    /// ultimately be inferred to be by-reference.
    ///
    /// Note that implicit reference skipping is not implemented yet (#42640).
    Unannotated,

    /// Annotated with `mut x` -- could be either ref or not, similar to `None`.
    Mutable,

    /// Annotated as `ref`, like `ref x`
    Ref,

    /// Annotated as `ref mut x`.
    RefMut,
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum RangeEnd {
    Included,
    Excluded,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum PatKind<'a> {
    /// Represents a wildcard pattern (`_`)
    Wild,

    /// A fresh binding `ref mut binding @ OPT_SUBPATTERN`.
    /// The `NodeId` is the canonical ID for the variable being bound,
    /// e.g., in `Ok(x) | Err(x)`, both `x` use the same canonical ID,
    /// which is the pattern ID of the first `x`.
    Binding(BindingAnnotation, NodeId, Ident, Option<P<'a, Pat<'a>>>),

    /// A struct or struct variant pattern, e.g., `Variant {x, y, ..}`.
    /// The `bool` is `true` in the presence of a `..`.
    Struct(QPath<'a>, HirVec<'a, Spanned<FieldPat<'a>>>, bool),

    /// A tuple struct/variant pattern `Variant(x, y, .., z)`.
    /// If the `..` pattern fragment is present, then `Option<usize>` denotes its position.
    /// 0 <= position <= subpats.len()
    TupleStruct(QPath<'a>, HirVec<'a, P<'a, Pat<'a>>>, Option<usize>),

    /// A path pattern for an unit struct/variant or a (maybe-associated) constant.
    Path(QPath<'a>),

    /// A tuple pattern `(a, b)`.
    /// If the `..` pattern fragment is present, then `Option<usize>` denotes its position.
    /// 0 <= position <= subpats.len()
    Tuple(HirVec<'a, P<'a, Pat<'a>>>, Option<usize>),
    /// A `box` pattern
    Box(P<'a, Pat<'a>>),
    /// A reference pattern, e.g., `&mut (a, b)`
    Ref(P<'a, Pat<'a>>, Mutability),
    /// A literal
    Lit(P<'a, Expr<'a>>),
    /// A range pattern, e.g., `1...2` or `1..2`
    Range(P<'a, Expr<'a>>, P<'a, Expr<'a>>, RangeEnd),
    /// `[a, b, ..i, y, z]` is represented as:
    ///     `PatKind::Slice(box [a, b], Some(i), box [y, z])`
    Slice(HirVec<'a, P<'a, Pat<'a>>>, Option<P<'a, Pat<'a>>>, HirVec<'a, P<'a, Pat<'a>>>),
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, HashStable,
         RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum Mutability {
    MutMutable,
    MutImmutable,
}

impl Mutability {
    /// Return MutMutable only if both arguments are mutable.
    pub fn and(self, other: Self) -> Self {
        match self {
            MutMutable => other,
            MutImmutable => MutImmutable,
        }
    }
}

#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, Copy, Hash, HashStable)]
pub enum BinOpKind {
    /// The `+` operator (addition)
    Add,
    /// The `-` operator (subtraction)
    Sub,
    /// The `*` operator (multiplication)
    Mul,
    /// The `/` operator (division)
    Div,
    /// The `%` operator (modulus)
    Rem,
    /// The `&&` operator (logical and)
    And,
    /// The `||` operator (logical or)
    Or,
    /// The `^` operator (bitwise xor)
    BitXor,
    /// The `&` operator (bitwise and)
    BitAnd,
    /// The `|` operator (bitwise or)
    BitOr,
    /// The `<<` operator (shift left)
    Shl,
    /// The `>>` operator (shift right)
    Shr,
    /// The `==` operator (equality)
    Eq,
    /// The `<` operator (less than)
    Lt,
    /// The `<=` operator (less than or equal to)
    Le,
    /// The `!=` operator (not equal to)
    Ne,
    /// The `>=` operator (greater than or equal to)
    Ge,
    /// The `>` operator (greater than)
    Gt,
}

impl BinOpKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BinOpKind::Add => "+",
            BinOpKind::Sub => "-",
            BinOpKind::Mul => "*",
            BinOpKind::Div => "/",
            BinOpKind::Rem => "%",
            BinOpKind::And => "&&",
            BinOpKind::Or => "||",
            BinOpKind::BitXor => "^",
            BinOpKind::BitAnd => "&",
            BinOpKind::BitOr => "|",
            BinOpKind::Shl => "<<",
            BinOpKind::Shr => ">>",
            BinOpKind::Eq => "==",
            BinOpKind::Lt => "<",
            BinOpKind::Le => "<=",
            BinOpKind::Ne => "!=",
            BinOpKind::Ge => ">=",
            BinOpKind::Gt => ">",
        }
    }

    pub fn is_lazy(self) -> bool {
        match self {
            BinOpKind::And | BinOpKind::Or => true,
            _ => false,
        }
    }

    pub fn is_shift(self) -> bool {
        match self {
            BinOpKind::Shl | BinOpKind::Shr => true,
            _ => false,
        }
    }

    pub fn is_comparison(self) -> bool {
        match self {
            BinOpKind::Eq |
            BinOpKind::Lt |
            BinOpKind::Le |
            BinOpKind::Ne |
            BinOpKind::Gt |
            BinOpKind::Ge => true,
            BinOpKind::And |
            BinOpKind::Or |
            BinOpKind::Add |
            BinOpKind::Sub |
            BinOpKind::Mul |
            BinOpKind::Div |
            BinOpKind::Rem |
            BinOpKind::BitXor |
            BinOpKind::BitAnd |
            BinOpKind::BitOr |
            BinOpKind::Shl |
            BinOpKind::Shr => false,
        }
    }

    /// Returns `true` if the binary operator takes its arguments by value
    pub fn is_by_value(self) -> bool {
        !self.is_comparison()
    }
}

impl Into<ast::BinOpKind> for BinOpKind {
    fn into(self) -> ast::BinOpKind {
        match self {
            BinOpKind::Add => ast::BinOpKind::Add,
            BinOpKind::Sub => ast::BinOpKind::Sub,
            BinOpKind::Mul => ast::BinOpKind::Mul,
            BinOpKind::Div => ast::BinOpKind::Div,
            BinOpKind::Rem => ast::BinOpKind::Rem,
            BinOpKind::And => ast::BinOpKind::And,
            BinOpKind::Or => ast::BinOpKind::Or,
            BinOpKind::BitXor => ast::BinOpKind::BitXor,
            BinOpKind::BitAnd => ast::BinOpKind::BitAnd,
            BinOpKind::BitOr => ast::BinOpKind::BitOr,
            BinOpKind::Shl => ast::BinOpKind::Shl,
            BinOpKind::Shr => ast::BinOpKind::Shr,
            BinOpKind::Eq => ast::BinOpKind::Eq,
            BinOpKind::Lt => ast::BinOpKind::Lt,
            BinOpKind::Le => ast::BinOpKind::Le,
            BinOpKind::Ne => ast::BinOpKind::Ne,
            BinOpKind::Ge => ast::BinOpKind::Ge,
            BinOpKind::Gt => ast::BinOpKind::Gt,
        }
    }
}

pub type BinOp = Spanned<BinOpKind>;

#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, Copy, Hash, HashStable)]
pub enum UnOp {
    /// The `*` operator for dereferencing
    UnDeref,
    /// The `!` operator for logical inversion
    UnNot,
    /// The `-` operator for negation
    UnNeg,
}

impl UnOp {
    pub fn as_str(self) -> &'static str {
        match self {
            UnDeref => "*",
            UnNot => "!",
            UnNeg => "-",
        }
    }

    /// Returns `true` if the unary operator takes its argument by value
    pub fn is_by_value(self) -> bool {
        match self {
            UnNeg | UnNot => true,
            _ => false,
        }
    }
}

/// A statement
pub type Stmt<'a> = Spanned<StmtKind<'a>>;

impl fmt::Debug for StmtKind<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Sadness.
        let spanned = source_map::dummy_spanned(self.clone());
        write!(f,
               "stmt({}: {})",
               spanned.node.id(),
               print::to_string(print::NO_ANN, |s| s.print_stmt(&spanned)))
    }
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, HashStable)]
pub enum StmtKind<'a> {
    /// Could be an item or a local (let) binding:
    Decl(P<'a, Decl<'a>>, NodeId),

    /// Expr without trailing semi-colon (must have unit type):
    Expr(P<'a, Expr<'a>>, NodeId),

    /// Expr with trailing semi-colon (may have any type):
    Semi(P<'a, Expr<'a>>, NodeId),
}

impl StmtKind<'_> {
    pub fn attrs(&self) -> &[Attribute] {
        match *self {
            StmtKind::Decl(ref d, _) => d.node.attrs(),
            StmtKind::Expr(ref e, _) |
            StmtKind::Semi(ref e, _) => &e.attrs,
        }
    }

    pub fn id(&self) -> NodeId {
        match *self {
            StmtKind::Decl(_, id) |
            StmtKind::Expr(_, id) |
            StmtKind::Semi(_, id) => id,
        }
    }
}

/// Local represents a `let` statement, e.g., `let <pat>:<ty> = <expr>;`
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct Local<'a> {
    pub pat: P<'a, Pat<'a>>,
    pub ty: Option<P<'a, Ty<'a>>>,
    /// Initializer expression to set the value, if any
    pub init: Option<P<'a, Expr<'a>>>,
    pub id: NodeId,
    pub hir_id: HirId,
    pub span: Span,
    pub attrs: ThinHirVec<'a, Attribute>,
    pub source: LocalSource,
}

pub type Decl<'a> = Spanned<DeclKind<'a>>;

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum DeclKind<'a> {
    /// A local (let) binding:
    Local(P<'a, Local<'a>>),
    /// An item binding:
    Item(ItemId),
}

impl DeclKind<'_> {
    pub fn attrs(&self) -> &[Attribute] {
        match *self {
            DeclKind::Local(ref l) => &l.attrs,
            DeclKind::Item(_) => &[]
        }
    }

    pub fn is_local(&self) -> bool {
        match *self {
            DeclKind::Local(_) => true,
            _ => false,
        }
    }
}

/// represents one arm of a 'match'
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct Arm<'a> {
    pub attrs: HirVec<'a, Attribute>,
    pub pats: HirVec<'a, P<'a, Pat<'a>>>,
    pub guard: Option<Guard<'a>>,
    pub body: P<'a, Expr<'a>>,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum Guard<'a> {
    If(P<'a, Expr<'a>>),
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct Field<'a> {
    #[stable_hasher(ignore)]
    pub id: NodeId,
    pub ident: Ident,
    pub expr: P<'a, Expr<'a>>,
    pub span: Span,
    pub is_shorthand: bool,
}

#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, Copy, HashStable)]
pub enum BlockCheckMode {
    DefaultBlock,
    UnsafeBlock(UnsafeSource),
    PushUnsafeBlock(UnsafeSource),
    PopUnsafeBlock(UnsafeSource),
}

#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, Copy, HashStable)]
pub enum UnsafeSource {
    CompilerGenerated,
    UserProvided,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct BodyId {
    pub node_id: NodeId,
}

/// The body of a function, closure, or constant value. In the case of
/// a function, the body contains not only the function body itself
/// (which is an expression), but also the argument patterns, since
/// those are something that the caller doesn't really care about.
///
/// # Examples
///
/// ```
/// fn foo((x, y): (u32, u32)) -> u32 {
///     x + y
/// }
/// ```
///
/// Here, the `Body` associated with `foo()` would contain:
///
/// - an `arguments` array containing the `(x, y)` pattern
/// - a `value` containing the `x + y` expression (maybe wrapped in a block)
/// - `is_generator` would be false
///
/// All bodies have an **owner**, which can be accessed via the HIR
/// map using `body_owner_def_id()`.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Body<'a> {
    pub arguments: HirVec<'a, Arg<'a>>,
    pub value: Expr<'a>,
    pub is_generator: bool,
}

impl Body<'_> {
    pub fn id(&self) -> BodyId {
        BodyId {
            node_id: self.value.id
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum BodyOwnerKind {
    /// Functions and methods.
    Fn,

    /// Constants and associated constants.
    Const,

    /// Initializer of a `static` item.
    Static(Mutability),
}

/// A constant (expression) that's not an item or associated item,
/// but needs its own `DefId` for type-checking, const-eval, etc.
/// These are usually found nested inside types (e.g., array lengths)
/// or expressions (e.g., repeat counts), and also used to define
/// explicit discriminant values for enum variants.
#[derive(Copy, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct AnonConst {
    pub id: NodeId,
    pub hir_id: HirId,
    pub body: BodyId,
}

/// An expression
#[derive(Copy, Clone, RustcEncodable, RustcDecodable)]
pub struct Expr<'a> {
    pub id: NodeId,
    pub span: Span,
    pub node: ExprKind<'a>,
    pub attrs: ThinHirVec<'a, Attribute>,
    pub hir_id: HirId,
}

impl Expr<'_> {
    pub fn precedence(&self) -> ExprPrecedence {
        match self.node {
            ExprKind::Box(_) => ExprPrecedence::Box,
            ExprKind::Array(_) => ExprPrecedence::Array,
            ExprKind::Call(..) => ExprPrecedence::Call,
            ExprKind::MethodCall(..) => ExprPrecedence::MethodCall,
            ExprKind::Tup(_) => ExprPrecedence::Tup,
            ExprKind::Binary(op, ..) => ExprPrecedence::Binary(op.node.into()),
            ExprKind::Unary(..) => ExprPrecedence::Unary,
            ExprKind::Lit(_) => ExprPrecedence::Lit,
            ExprKind::Type(..) | ExprKind::Cast(..) => ExprPrecedence::Cast,
            ExprKind::If(..) => ExprPrecedence::If,
            ExprKind::While(..) => ExprPrecedence::While,
            ExprKind::Loop(..) => ExprPrecedence::Loop,
            ExprKind::Match(..) => ExprPrecedence::Match,
            ExprKind::Closure(..) => ExprPrecedence::Closure,
            ExprKind::Block(..) => ExprPrecedence::Block,
            ExprKind::Assign(..) => ExprPrecedence::Assign,
            ExprKind::AssignOp(..) => ExprPrecedence::AssignOp,
            ExprKind::Field(..) => ExprPrecedence::Field,
            ExprKind::Index(..) => ExprPrecedence::Index,
            ExprKind::Path(..) => ExprPrecedence::Path,
            ExprKind::AddrOf(..) => ExprPrecedence::AddrOf,
            ExprKind::Break(..) => ExprPrecedence::Break,
            ExprKind::Continue(..) => ExprPrecedence::Continue,
            ExprKind::Ret(..) => ExprPrecedence::Ret,
            ExprKind::InlineAsm(..) => ExprPrecedence::InlineAsm,
            ExprKind::Struct(..) => ExprPrecedence::Struct,
            ExprKind::Repeat(..) => ExprPrecedence::Repeat,
            ExprKind::Yield(..) => ExprPrecedence::Yield,
            ExprKind::Err => ExprPrecedence::Err,
        }
    }

    pub fn is_place_expr(&self) -> bool {
         match self.node {
            ExprKind::Path(QPath::Resolved(_, ref path)) => {
                match path.def {
                    Def::Local(..) | Def::Upvar(..) | Def::Static(..) | Def::Err => true,
                    _ => false,
                }
            }

            ExprKind::Type(ref e, _) => {
                e.is_place_expr()
            }

            ExprKind::Unary(UnDeref, _) |
            ExprKind::Field(..) |
            ExprKind::Index(..) => {
                true
            }

            // Partially qualified paths in expressions can only legally
            // refer to associated items which are always rvalues.
            ExprKind::Path(QPath::TypeRelative(..)) |

            ExprKind::Call(..) |
            ExprKind::MethodCall(..) |
            ExprKind::Struct(..) |
            ExprKind::Tup(..) |
            ExprKind::If(..) |
            ExprKind::Match(..) |
            ExprKind::Closure(..) |
            ExprKind::Block(..) |
            ExprKind::Repeat(..) |
            ExprKind::Array(..) |
            ExprKind::Break(..) |
            ExprKind::Continue(..) |
            ExprKind::Ret(..) |
            ExprKind::While(..) |
            ExprKind::Loop(..) |
            ExprKind::Assign(..) |
            ExprKind::InlineAsm(..) |
            ExprKind::AssignOp(..) |
            ExprKind::Lit(_) |
            ExprKind::Unary(..) |
            ExprKind::Box(..) |
            ExprKind::AddrOf(..) |
            ExprKind::Binary(..) |
            ExprKind::Yield(..) |
            ExprKind::Cast(..) |
            ExprKind::Err => {
                false
            }
        }
    }
}

impl fmt::Debug for Expr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "expr({}: {})", self.id,
               print::to_string(print::NO_ANN, |s| s.print_expr(self)))
    }
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum ExprKind<'a> {
    /// A `box x` expression.
    Box(P<'a, Expr<'a>>),
    /// An array (`[a, b, c, d]`)
    Array(HirVec<'a, Expr<'a>>),
    /// A function call
    ///
    /// The first field resolves to the function itself (usually an `ExprKind::Path`),
    /// and the second field is the list of arguments.
    /// This also represents calling the constructor of
    /// tuple-like ADTs such as tuple structs and enum variants.
    Call(P<'a, Expr<'a>>, HirVec<'a, Expr<'a>>),
    /// A method call (`x.foo::<'static, Bar, Baz>(a, b, c, d)`)
    ///
    /// The `PathSegment`/`Span` represent the method name and its generic arguments
    /// (within the angle brackets).
    /// The first element of the vector of `Expr`s is the expression that evaluates
    /// to the object on which the method is being called on (the receiver),
    /// and the remaining elements are the rest of the arguments.
    /// Thus, `x.foo::<Bar, Baz>(a, b, c, d)` is represented as
    /// `ExprKind::MethodCall(PathSegment { foo, [Bar, Baz] }, [x, a, b, c, d])`.
    MethodCall(PathSegment<'a>, Span, HirVec<'a, Expr<'a>>),
    /// A tuple (`(a, b, c ,d)`)
    Tup(HirVec<'a, Expr<'a>>),
    /// A binary operation (For example: `a + b`, `a * b`)
    Binary(BinOp, P<'a, Expr<'a>>, P<'a, Expr<'a>>),
    /// A unary operation (For example: `!x`, `*x`)
    Unary(UnOp, P<'a, Expr<'a>>),
    /// A literal (For example: `1`, `"foo"`)
    Lit(P<'a, Lit>),
    /// A cast (`foo as f64`)
    Cast(P<'a, Expr<'a>>, P<'a, Ty<'a>>),
    Type(P<'a, Expr<'a>>, P<'a, Ty<'a>>),
    /// An `if` block, with an optional else block
    ///
    /// `if expr { expr } else { expr }`
    If(P<'a, Expr<'a>>, P<'a, Expr<'a>>, Option<P<'a, Expr<'a>>>),
    /// A while loop, with an optional label
    ///
    /// `'label: while expr { block }`
    While(P<'a, Expr<'a>>, P<'a, Block<'a>>, Option<Label>),
    /// Conditionless loop (can be exited with break, continue, or return)
    ///
    /// `'label: loop { block }`
    Loop(P<'a, Block<'a>>, Option<Label>, LoopSource),
    /// A `match` block, with a source that indicates whether or not it is
    /// the result of a desugaring, and if so, which kind.
    Match(P<'a, Expr<'a>>, HirVec<'a, Arm<'a>>, MatchSource),
    /// A closure (for example, `move |a, b, c| {a + b + c}`).
    ///
    /// The final span is the span of the argument block `|...|`
    ///
    /// This may also be a generator literal, indicated by the final boolean,
    /// in that case there is an GeneratorClause.
    Closure(CaptureClause, P<'a, FnDecl<'a>>, BodyId, Span, Option<GeneratorMovability>),
    /// A block (`'label: { ... }`)
    Block(P<'a, Block<'a>>, Option<Label>),

    /// An assignment (`a = foo()`)
    Assign(P<'a, Expr<'a>>, P<'a, Expr<'a>>),
    /// An assignment with an operator
    ///
    /// For example, `a += 1`.
    AssignOp(BinOp, P<'a, Expr<'a>>, P<'a, Expr<'a>>),
    /// Access of a named (`obj.foo`) or unnamed (`obj.0`) struct or tuple field
    Field(P<'a, Expr<'a>>, Ident),
    /// An indexing operation (`foo[2]`)
    Index(P<'a, Expr<'a>>, P<'a, Expr<'a>>),

    /// Path to a definition, possibly containing lifetime or type parameters.
    Path(QPath<'a>),

    /// A referencing operation (`&a` or `&mut a`)
    AddrOf(Mutability, P<'a, Expr<'a>>),
    /// A `break`, with an optional label to break
    Break(Destination, Option<P<'a, Expr<'a>>>),
    /// A `continue`, with an optional label
    Continue(Destination),
    /// A `return`, with an optional value to be returned
    Ret(Option<P<'a, Expr<'a>>>),

    /// Inline assembly (from `asm!`), with its outputs and inputs.
    InlineAsm(P<'a, InlineAsm>, HirVec<'a, Expr<'a>>, HirVec<'a, Expr<'a>>),

    /// A struct or struct-like variant literal expression.
    ///
    /// For example, `Foo {x: 1, y: 2}`, or
    /// `Foo {x: 1, .. base}`, where `base` is the `Option<Expr>`.
    Struct(QPath<'a>, HirVec<'a, Field<'a>>, Option<P<'a, Expr<'a>>>),

    /// An array literal constructed from one repeated element.
    ///
    /// For example, `[1; 5]`. The first expression is the element
    /// to be repeated; the second is the number of times to repeat it.
    Repeat(P<'a, Expr<'a>>, AnonConst),

    /// A suspension point for generators. This is `yield <expr>` in Rust.
    Yield(P<'a, Expr<'a>>),

    /// Placeholder for an expression that wasn't syntactically well formed in some way.
    Err,
}

/// Optionally `Self`-qualified value/type path or associated extension.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum QPath<'a> {
    /// Path to a definition, optionally "fully-qualified" with a `Self`
    /// type, if the path points to an associated item in a trait.
    ///
    /// e.g., an unqualified path like `Clone::clone` has `None` for `Self`,
    /// while `<Vec<T> as Clone>::clone` has `Some(Vec<T>)` for `Self`,
    /// even though they both have the same two-segment `Clone::clone` `Path`.
    Resolved(Option<P<'a, Ty<'a>>>, P<'a, Path<'a>>),

    /// Type-related paths, e.g., `<T>::default` or `<T>::Output`.
    /// Will be resolved by type-checking to an associated item.
    ///
    /// UFCS source paths can desugar into this, with `Vec::new` turning into
    /// `<Vec>::new`, and `T::X::Y::method` into `<<<T>::X>::Y>::method`,
    /// the `X` and `Y` nodes each being a `TyKind::Path(QPath<'a>::TypeRelative(..))`.
    TypeRelative(P<'a, Ty<'a>>, P<'a, PathSegment<'a>>)
}

/// Hints at the original code for a let statement
#[derive(Clone, RustcEncodable, RustcDecodable, Debug, Copy, HashStable)]
pub enum LocalSource {
    /// A `match _ { .. }`
    Normal,
    /// A desugared `for _ in _ { .. }` loop
    ForLoopDesugar,
}

/// Hints at the original code for a `match _ { .. }`
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy, HashStable)]
pub enum MatchSource {
    /// A `match _ { .. }`
    Normal,
    /// An `if let _ = _ { .. }` (optionally with `else { .. }`)
    IfLetDesugar {
        contains_else_clause: bool,
    },
    /// A `while let _ = _ { .. }` (which was desugared to a
    /// `loop { match _ { .. } }`)
    WhileLetDesugar,
    /// A desugared `for _ in _ { .. }` loop
    ForLoopDesugar,
    /// A desugared `?` operator
    TryDesugar,
}

/// The loop type that yielded an ExprKind::Loop
#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, Copy, HashStable)]
pub enum LoopSource {
    /// A `loop { .. }` loop
    Loop,
    /// A `while let _ = _ { .. }` loop
    WhileLet,
    /// A `for _ in _ { .. }` loop
    ForLoop,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug, Copy, HashStable)]
pub enum LoopIdError {
    OutsideLoopScope,
    UnlabeledCfInWhileCondition,
    UnresolvedLabel,
}

impl fmt::Display for LoopIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(match *self {
            LoopIdError::OutsideLoopScope => "not inside loop scope",
            LoopIdError::UnlabeledCfInWhileCondition =>
                "unlabeled control flow (break or continue) in while condition",
            LoopIdError::UnresolvedLabel => "label not found",
        }, f)
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug, Copy, HashStable)]
pub struct Destination {
    // This is `Some(_)` iff there is an explicit user-specified `label
    pub label: Option<Label>,

    // These errors are caught and then reported during the diagnostics pass in
    // librustc_passes/loops.rs
    pub target_id: Result<NodeId, LoopIdError>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, HashStable,
         RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum GeneratorMovability {
    Static,
    Movable,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug, Copy, HashStable)]
pub enum CaptureClause {
    CaptureByValue,
    CaptureByRef,
}

// N.B., if you change this, you'll probably want to change the corresponding
// type structure in middle/ty.rs as well.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct MutTy<'a> {
    pub ty: P<'a, Ty<'a>>,
    pub mutbl: Mutability,
}

/// Represents a method's signature in a trait declaration or implementation.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct MethodSig<'a> {
    pub header: FnHeader,
    pub decl: P<'a, FnDecl<'a>>,
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the node-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Debug)]
pub struct TraitItemId {
    pub node_id: NodeId,
}

/// Represents an item declaration within a trait declaration,
/// possibly including a default implementation. A trait item is
/// either required (meaning it doesn't have an implementation, just a
/// signature) or provided (meaning it has a default implementation).
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct TraitItem<'a> {
    #[stable_hasher(ignore)]
    pub id: NodeId,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub attrs: HirVec<'a, Attribute>,
    pub generics: Generics<'a>,
    pub node: TraitItemKind<'a>,
    pub span: Span,
}

/// A trait method's body (or just argument names).
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum TraitMethod<'a> {
    /// No default body in the trait, just a signature.
    Required(HirVec<'a, Ident>),

    /// Both signature and body are provided in the trait.
    Provided(BodyId),
}

/// Represents a trait method or associated constant or type
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum TraitItemKind<'a> {
    /// An associated constant with an optional value (otherwise `impl`s
    /// must contain a value)
    Const(P<'a, Ty<'a>>, Option<BodyId>),
    /// A method with an optional body
    Method(MethodSig<'a>, TraitMethod<'a>),
    /// An associated type with (possibly empty) bounds and optional concrete
    /// type
    Type(GenericBounds<'a>, Option<P<'a, Ty<'a>>>),
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the node-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Debug)]
pub struct ImplItemId {
    pub node_id: NodeId,
}

/// Represents anything within an `impl` block
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct ImplItem<'a> {
    #[stable_hasher(ignore)]
    pub id: NodeId,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub vis: Visibility<'a>,
    pub defaultness: Defaultness,
    pub attrs: HirVec<'a, Attribute>,
    pub generics: Generics<'a>,
    pub node: ImplItemKind<'a>,
    pub span: Span,
}

/// Represents different contents within `impl`s
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum ImplItemKind<'a> {
    /// An associated constant of the given type, set to the constant result
    /// of the expression
    Const(P<'a, Ty<'a>>, BodyId),
    /// A method implementation with the given signature and body
    Method(MethodSig<'a>, BodyId),
    /// An associated type
    Type(P<'a, Ty<'a>>),
    /// An associated existential type
    Existential(GenericBounds<'a>),
}

// Bind a type to an associated type: `A=Foo`.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct TypeBinding<'a> {
    pub id: NodeId,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub ty: P<'a, Ty<'a>>,
    pub span: Span,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable)]
pub struct Ty<'a> {
    pub id: NodeId,
    pub node: TyKind<'a>,
    pub span: Span,
    pub hir_id: HirId,
}

impl fmt::Debug for Ty<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type({})",
               print::to_string(print::NO_ANN, |s| s.print_type(self)))
    }
}

/// Not represented directly in the AST, referred to by name through a ty_path.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy, HashStable)]
pub enum PrimTy {
    Int(IntTy),
    Uint(UintTy),
    Float(FloatTy),
    Str,
    Bool,
    Char,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct BareFnTy<'a> {
    pub unsafety: Unsafety,
    pub abi: Abi,
    pub generic_params: HirVec<'a, GenericParam<'a>>,
    pub decl: P<'a, FnDecl<'a>>,
    pub arg_names: HirVec<'a, Ident>,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct ExistTy<'a> {
    pub generics: Generics<'a>,
    pub bounds: GenericBounds<'a>,
    pub impl_trait_fn: Option<DefId>,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
/// The different kinds of types recognized by the compiler
pub enum TyKind<'a> {
    /// A variable length slice (`[T]`)
    Slice(P<'a, Ty<'a>>),
    /// A fixed length array (`[T; n]`)
    Array(P<'a, Ty<'a>>, AnonConst),
    /// A raw pointer (`*const T` or `*mut T`)
    Ptr(MutTy<'a>),
    /// A reference (`&'a T` or `&'a mut T`)
    Rptr(Lifetime, MutTy<'a>),
    /// A bare function (e.g., `fn(usize) -> bool`)
    BareFn(P<'a, BareFnTy<'a>>),
    /// The never type (`!`)
    Never,
    /// A tuple (`(A, B, C, D,...)`)
    Tup(HirVec<'a, Ty<'a>>),
    /// A path to a type definition (`module::module::...::Type`), or an
    /// associated type, e.g., `<Vec<T> as Trait>::Type` or `<T>::Target`.
    ///
    /// Type parameters may be stored in each `PathSegment`.
    Path(QPath<'a>),
    /// A type definition itself. This is currently only used for the `existential type`
    /// item that `impl Trait` in return position desugars to.
    ///
    /// The generic arg list are the lifetimes (and in the future possibly parameters) that are
    /// actually bound on the `impl Trait`.
    Def(ItemId, HirVec<'a, GenericArg<'a>>),
    /// A trait object type `Bound1 + Bound2 + Bound3`
    /// where `Bound` is a trait or a lifetime.
    TraitObject(HirVec<'a, PolyTraitRef<'a>>, Lifetime),
    /// Unused for now
    Typeof(AnonConst),
    /// `TyKind::Infer` means the type should be inferred instead of it having been
    /// specified. This can appear anywhere in a type.
    Infer,
    /// Placeholder for a type that has failed to be defined.
    Err,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct InlineAsmOutput {
    pub constraint: Symbol,
    pub is_rw: bool,
    pub is_indirect: bool,
    pub span: Span,
}

#[derive(Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct InlineAsm {
    pub asm: Symbol,
    pub asm_str_style: StrStyle,
    pub outputs: Vec<InlineAsmOutput>,
    pub inputs: Vec<Symbol>,
    pub clobbers: Vec<Symbol>,
    pub volatile: bool,
    pub alignstack: bool,
    pub dialect: AsmDialect,
    #[stable_hasher(ignore)] // This is used for error reporting
    pub ctxt: SyntaxContext,
}

/// represents an argument in a function header
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct Arg<'a> {
    pub pat: P<'a, Pat<'a>>,
    pub id: NodeId,
    pub hir_id: HirId,
}

/// Represents the header (not the body) of a function declaration
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct FnDecl<'a> {
    pub inputs: HirVec<'a, Ty<'a>>,
    pub output: FunctionRetTy<'a>,
    pub variadic: bool,
    /// Does the function have an implicit self?
    pub implicit_self: ImplicitSelfKind,
}

/// Represents what type of implicit self a function has, if any.
#[derive(Clone, Copy, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum ImplicitSelfKind {
    /// Represents a `fn x(self);`.
    Imm,
    /// Represents a `fn x(mut self);`.
    Mut,
    /// Represents a `fn x(&self);`.
    ImmRef,
    /// Represents a `fn x(&mut self);`.
    MutRef,
    /// Represents when a function does not have a self argument or
    /// when a function has a `self: X` argument.
    None
}

impl ImplicitSelfKind {
    /// Does this represent an implicit self?
    pub fn has_implicit_self(&self) -> bool {
        match *self {
            ImplicitSelfKind::None => false,
            _ => true,
        }
    }
}

/// Is the trait definition an auto trait?
#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum IsAuto {
    Yes,
    No
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, HashStable,
         Ord, RustcEncodable, RustcDecodable, Debug)]
pub enum IsAsync {
    Async,
    NotAsync,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, HashStable,
         RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Unsafety {
    Unsafe,
    Normal,
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum Constness {
    Const,
    NotConst,
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum Defaultness {
    Default { has_value: bool },
    Final,
}

impl Defaultness {
    pub fn has_value(&self) -> bool {
        match *self {
            Defaultness::Default { has_value, .. } => has_value,
            Defaultness::Final => true,
        }
    }

    pub fn is_final(&self) -> bool {
        *self == Defaultness::Final
    }

    pub fn is_default(&self) -> bool {
        match *self {
            Defaultness::Default { .. } => true,
            _ => false,
        }
    }
}

impl fmt::Display for Unsafety {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(match *self {
                              Unsafety::Normal => "normal",
                              Unsafety::Unsafe => "unsafe",
                          },
                          f)
    }
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, HashStable)]
pub enum ImplPolarity {
    /// `impl Trait for Type`
    Positive,
    /// `impl !Trait for Type`
    Negative,
}

impl fmt::Debug for ImplPolarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ImplPolarity::Positive => "positive".fmt(f),
            ImplPolarity::Negative => "negative".fmt(f),
        }
    }
}


#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum FunctionRetTy<'a> {
    /// Return type is not specified.
    ///
    /// Functions default to `()` and
    /// closures default to inference. Span points to where return
    /// type would be inserted.
    DefaultReturn(Span),
    /// Everything else
    Return(P<'a, Ty<'a>>),
}

impl fmt::Display for FunctionRetTy<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Return(ref ty) => print::to_string(print::NO_ANN, |s| s.print_type(ty)).fmt(f),
            DefaultReturn(_) => "()".fmt(f),
        }
    }
}

impl FunctionRetTy<'_> {
    pub fn span(&self) -> Span {
        match *self {
            DefaultReturn(span) => span,
            Return(ref ty) => ty.span,
        }
    }
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct Mod<'a> {
    /// A span from the first token past `{` to the last token until `}`.
    /// For `mod foo;`, the inner span ranges from the first token
    /// to the last token in the external file.
    pub inner: Span,
    pub item_ids: HirVec<'a, ItemId>,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct ForeignMod<'a> {
    pub abi: Abi,
    pub items: HirVec<'a, ForeignItem<'a>>,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct GlobalAsm {
    pub asm: Symbol,
    #[stable_hasher(ignore)] // This is used for error reporting
    pub ctxt: SyntaxContext,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct EnumDef<'a> {
    pub variants: HirVec<'a, Variant<'a>>,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct VariantKind<'a> {
    pub ident: Ident,
    pub attrs: HirVec<'a, Attribute>,
    pub data: VariantData<'a>,
    /// Explicit discriminant, e.g., `Foo = 1`
    pub disr_expr: Option<AnonConst>,
}

pub type Variant<'a> = Spanned<VariantKind<'a>>;

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum UseKind {
    /// One import, e.g., `use foo::bar` or `use foo::bar as baz`.
    /// Also produced for each element of a list `use`, e.g.
    // `use foo::{a, b}` lowers to `use foo::a; use foo::b;`.
    Single,

    /// Glob import, e.g., `use foo::*`.
    Glob,

    /// Degenerate list import, e.g., `use foo::{a, b}` produces
    /// an additional `use foo::{}` for performing checks such as
    /// unstable feature gating. May be removed in the future.
    ListStem,
}

/// TraitRef's appear in impls.
///
/// resolve maps each TraitRef's ref_id to its defining trait; that's all
/// that the ref_id is for. Note that ref_id's value is not the NodeId of the
/// trait being referred to but just a unique NodeId that serves as a key
/// within the DefMap.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct TraitRef<'a> {
    pub path: Path<'a>,
    // Don't hash the ref_id. It is tracked via the thing it is used to access
    #[stable_hasher(ignore)]
    pub ref_id: NodeId,
    #[stable_hasher(ignore)]
    pub hir_ref_id: HirId,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct PolyTraitRef<'a> {
    /// The `'a` in `<'a> Foo<&'a T>`
    pub bound_generic_params: HirVec<'a, GenericParam<'a>>,

    /// The `Foo<&'a T>` in `<'a> Foo<&'a T>`
    pub trait_ref: TraitRef<'a>,

    pub span: Span,
}

pub type Visibility<'a> = Spanned<VisibilityKind<'a>>;

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug)]
pub enum VisibilityKind<'a> {
    Public,
    Crate(CrateSugar),
    Restricted { path: P<'a, Path<'a>>, id: NodeId, hir_id: HirId },
    Inherited,
}

impl VisibilityKind<'_> {
    pub fn is_pub(&self) -> bool {
        match *self {
            VisibilityKind::Public => true,
            _ => false
        }
    }

    pub fn is_pub_restricted(&self) -> bool {
        match *self {
            VisibilityKind::Public |
            VisibilityKind::Inherited => false,
            VisibilityKind::Crate(..) |
            VisibilityKind::Restricted { .. } => true,
        }
    }

    pub fn descr(&self) -> &'static str {
        match *self {
            VisibilityKind::Public => "public",
            VisibilityKind::Inherited => "private",
            VisibilityKind::Crate(..) => "crate-visible",
            VisibilityKind::Restricted { .. } => "restricted",
        }
    }
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct StructField<'a> {
    pub span: Span,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub vis: Visibility<'a>,
    pub id: NodeId,
    pub ty: P<'a, Ty<'a>>,
    pub attrs: HirVec<'a, Attribute>,
}

impl StructField<'_> {
    // Still necessary in couple of places
    pub fn is_positional(&self) -> bool {
        let first = self.ident.as_str().as_bytes()[0];
        first >= b'0' && first <= b'9'
    }
}

/// Fields and Ids of enum variants and structs
///
/// For enum variants: `NodeId` represents both an Id of the variant itself (relevant for all
/// variant kinds) and an Id of the variant's constructor (not relevant for `Struct`-variants).
/// One shared Id can be successfully used for these two purposes.
/// Id of the whole enum lives in `Item`.
///
/// For structs: `NodeId` represents an Id of the structure's constructor, so it is not actually
/// used for `Struct`-structs (but still present). Structures don't have an analogue of "Id of
/// the variant itself" from enum variants.
/// Id of the whole struct lives in `Item`.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum VariantData<'a> {
    Struct(HirVec<'a, StructField<'a>>, NodeId),
    Tuple(HirVec<'a, StructField<'a>>, NodeId),
    Unit(NodeId),
}

impl<'a> VariantData<'a> {
    pub fn fields(&self) -> &[StructField<'a>] {
        match *self {
            VariantData::Struct(ref fields, _) | VariantData::Tuple(ref fields, _) => fields,
            _ => &[],
        }
    }
    pub fn id(&self) -> NodeId {
        match *self {
            VariantData::Struct(_, id) | VariantData::Tuple(_, id) | VariantData::Unit(id) => id,
        }
    }
    pub fn is_struct(&self) -> bool {
        if let VariantData::Struct(..) = *self {
            true
        } else {
            false
        }
    }
    pub fn is_tuple(&self) -> bool {
        if let VariantData::Tuple(..) = *self {
            true
        } else {
            false
        }
    }
    pub fn is_unit(&self) -> bool {
        if let VariantData::Unit(..) = *self {
            true
        } else {
            false
        }
    }
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the node-id of the item
// so it can fetched later.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ItemId {
    pub id: NodeId,
}

/// An item
///
/// The name might be a dummy name in case of anonymous items
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct Item<'a> {
    pub ident: Ident,
    #[stable_hasher(ignore)]
    pub id: NodeId,
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub attrs: HirVec<'a, Attribute>,
    pub node: ItemKind<'a>,
    pub vis: Visibility<'a>,
    pub span: Span,
}

#[derive(Clone, Copy, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct FnHeader {
    pub unsafety: Unsafety,
    pub constness: Constness,
    pub asyncness: IsAsync,
    pub abi: Abi,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum ItemKind<'a> {
    /// An `extern crate` item, with optional *original* crate name if the crate was renamed.
    ///
    /// e.g., `extern crate foo` or `extern crate foo_bar as foo`
    ExternCrate(Option<Name>),

    /// `use foo::bar::*;` or `use foo::bar::baz as quux;`
    ///
    /// or just
    ///
    /// `use foo::bar::baz;` (with `as baz` implicitly on the right)
    Use(P<'a, Path<'a>>, UseKind),

    /// A `static` item
    Static(P<'a, Ty<'a>>, Mutability, BodyId),
    /// A `const` item
    Const(P<'a, Ty<'a>>, BodyId),
    /// A function declaration
    Fn(P<'a, FnDecl<'a>>, FnHeader, Generics<'a>, BodyId),
    /// A module
    Mod(Mod<'a>),
    /// An external module
    ForeignMod(ForeignMod<'a>),
    /// Module-level inline assembly (from global_asm!)
    GlobalAsm(P<'a, GlobalAsm>),
    /// A type alias, e.g., `type Foo = Bar<u8>`
    Ty(P<'a, Ty<'a>>, Generics<'a>),
    /// An existential type definition, e.g., `existential type Foo: Bar;`
    Existential(ExistTy<'a>),
    /// An enum definition, e.g., `enum Foo<A, B> {C<A>, D<B>}`
    Enum(EnumDef<'a>, Generics<'a>),
    /// A struct definition, e.g., `struct Foo<A> {x: A}`
    Struct(VariantData<'a>, Generics<'a>),
    /// A union definition, e.g., `union Foo<A, B> {x: A, y: B}`
    Union(VariantData<'a>, Generics<'a>),
    /// Represents a Trait Declaration
    Trait(IsAuto, Unsafety, Generics<'a>, GenericBounds<'a>, HirVec<'a, TraitItemRef>),
    /// Represents a Trait Alias Declaration
    TraitAlias(Generics<'a>, GenericBounds<'a>),

    /// An implementation, eg `impl<A> Trait for Foo { .. }`
    Impl(Unsafety,
         ImplPolarity,
         Defaultness,
         Generics<'a>,
         Option<TraitRef<'a>>, // (optional) trait this impl implements
         P<'a, Ty<'a>>, // self
         HirVec<'a, ImplItemRef<'a>>),
}

impl ItemKind<'_> {
    pub fn descriptive_variant(&self) -> &str {
        match *self {
            ItemKind::ExternCrate(..) => "extern crate",
            ItemKind::Use(..) => "use",
            ItemKind::Static(..) => "static item",
            ItemKind::Const(..) => "constant item",
            ItemKind::Fn(..) => "function",
            ItemKind::Mod(..) => "module",
            ItemKind::ForeignMod(..) => "foreign module",
            ItemKind::GlobalAsm(..) => "global asm",
            ItemKind::Ty(..) => "type alias",
            ItemKind::Existential(..) => "existential type",
            ItemKind::Enum(..) => "enum",
            ItemKind::Struct(..) => "struct",
            ItemKind::Union(..) => "union",
            ItemKind::Trait(..) => "trait",
            ItemKind::TraitAlias(..) => "trait alias",
            ItemKind::Impl(..) => "item",
        }
    }

    pub fn adt_kind(&self) -> Option<AdtKind> {
        match *self {
            ItemKind::Struct(..) => Some(AdtKind::Struct),
            ItemKind::Union(..) => Some(AdtKind::Union),
            ItemKind::Enum(..) => Some(AdtKind::Enum),
            _ => None,
        }
    }

    pub fn generics(&self) -> Option<&Generics> {
        Some(match *self {
            ItemKind::Fn(_, _, ref generics, _) |
            ItemKind::Ty(_, ref generics) |
            ItemKind::Existential(ExistTy { ref generics, impl_trait_fn: None, .. }) |
            ItemKind::Enum(_, ref generics) |
            ItemKind::Struct(_, ref generics) |
            ItemKind::Union(_, ref generics) |
            ItemKind::Trait(_, _, ref generics, _, _) |
            ItemKind::Impl(_, _, _, ref generics, _, _, _)=> generics,
            _ => return None
        })
    }
}

/// A reference from an trait to one of its associated items. This
/// contains the item's id, naturally, but also the item's name and
/// some other high-level details (like whether it is an associated
/// type or method, and whether it is public). This allows other
/// passes to find the impl they want without loading the id (which
/// means fewer edges in the incremental compilation graph).
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct TraitItemRef {
    pub id: TraitItemId,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub kind: AssociatedItemKind,
    pub span: Span,
    pub defaultness: Defaultness,
}

/// A reference from an impl to one of its associated items. This
/// contains the item's id, naturally, but also the item's name and
/// some other high-level details (like whether it is an associated
/// type or method, and whether it is public). This allows other
/// passes to find the impl they want without loading the id (which
/// means fewer edges in the incremental compilation graph).
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct ImplItemRef<'a> {
    pub id: ImplItemId,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub kind: AssociatedItemKind,
    pub span: Span,
    pub vis: Visibility<'a>,
    pub defaultness: Defaultness,
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum AssociatedItemKind {
    Const,
    Method { has_self: bool },
    Type,
    Existential,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub struct ForeignItem<'a> {
    pub ident: Ident,
    pub attrs: HirVec<'a, Attribute>,
    pub node: ForeignItemKind<'a>,
    pub id: NodeId,
    pub span: Span,
    pub vis: Visibility<'a>,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ForeignItem2<'a> {
    pub name: Name,
    pub attrs: HirVec<'a, Attribute>,
    pub node: ForeignItemKind<'a>,
    pub id: NodeId,
    pub span: Span,
    pub vis: Visibility<'a>,
}

/// An item within an `extern` block
#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug, HashStable)]
pub enum ForeignItemKind<'a> {
    /// A foreign function
    Fn(P<'a, FnDecl<'a>>, HirVec<'a, Ident>, Generics<'a>),
    /// A foreign static item (`static ext: u8`), with optional mutability
    /// (the boolean is true when mutable)
    Static(P<'a, Ty<'a>>, bool),
    /// A foreign type
    Type,
}

impl ForeignItemKind<'_> {
    pub fn descriptive_variant(&self) -> &str {
        match *self {
            ForeignItemKind::Fn(..) => "foreign function",
            ForeignItemKind::Static(..) => "foreign static item",
            ForeignItemKind::Type => "foreign type",
        }
    }
}

/// A free variable referred to in a function.
#[derive(Debug, Copy, Clone, RustcEncodable, RustcDecodable, HashStable)]
pub struct Freevar {
    /// The variable being accessed free.
    pub def: Def,

    // First span where it is accessed (there can be multiple).
    pub span: Span
}

impl Freevar {
    pub fn var_id(&self) -> NodeId {
        match self.def {
            Def::Local(id) | Def::Upvar(id, ..) => id,
            _ => bug!("Freevar::var_id: bad def ({:?})", self.def)
        }
    }
}

pub type FreevarMap = NodeMap<Vec<Freevar>>;

pub type CaptureModeMap = NodeMap<CaptureClause>;

#[derive(Copy, Clone, Debug)]
pub struct TraitCandidate {
    pub def_id: DefId,
    pub import_id: Option<NodeId>,
}

// Trait method resolution
pub type TraitMap = NodeMap<Vec<TraitCandidate>>;

// Map from the NodeId of a glob import to a list of items which are actually
// imported.
pub type GlobMap = NodeMap<FxHashSet<Name>>;


pub fn provide(providers: &mut Providers<'_>) {
    providers.describe_def = map::describe_def;
}

#[derive(Clone, RustcEncodable, RustcDecodable, HashStable)]
pub struct CodegenFnAttrs {
    pub flags: CodegenFnAttrFlags,
    /// Parsed representation of the `#[inline]` attribute
    pub inline: InlineAttr,
    /// The `#[export_name = "..."]` attribute, indicating a custom symbol a
    /// function should be exported under
    pub export_name: Option<Symbol>,
    /// The `#[link_name = "..."]` attribute, indicating a custom symbol an
    /// imported function should be imported as. Note that `export_name`
    /// probably isn't set when this is set, this is for foreign items while
    /// `#[export_name]` is for Rust-defined functions.
    pub link_name: Option<Symbol>,
    /// The `#[target_feature(enable = "...")]` attribute and the enabled
    /// features (only enabled features are supported right now).
    pub target_features: Vec<Symbol>,
    /// The `#[linkage = "..."]` attribute and the value we found.
    pub linkage: Option<Linkage>,
    /// The `#[link_section = "..."]` attribute, or what executable section this
    /// should be placed in.
    pub link_section: Option<Symbol>,
}

bitflags! {
    #[derive(RustcEncodable, RustcDecodable, HashStable)]
    pub struct CodegenFnAttrFlags: u32 {
        /// #[cold], a hint to LLVM that this function, when called, is never on
        /// the hot path
        const COLD                      = 1 << 0;
        /// #[allocator], a hint to LLVM that the pointer returned from this
        /// function is never null
        const ALLOCATOR                 = 1 << 1;
        /// #[unwind], an indicator that this function may unwind despite what
        /// its ABI signature may otherwise imply
        const UNWIND                    = 1 << 2;
        /// #[rust_allocator_nounwind], an indicator that an imported FFI
        /// function will never unwind. Probably obsolete by recent changes with
        /// #[unwind], but hasn't been removed/migrated yet
        const RUSTC_ALLOCATOR_NOUNWIND  = 1 << 3;
        /// #[naked], indicates to LLVM that no function prologue/epilogue
        /// should be generated
        const NAKED                     = 1 << 4;
        /// #[no_mangle], the function's name should be the same as its symbol
        const NO_MANGLE                 = 1 << 5;
        /// #[rustc_std_internal_symbol], and indicator that this symbol is a
        /// "weird symbol" for the standard library in that it has slightly
        /// different linkage, visibility, and reachability rules.
        const RUSTC_STD_INTERNAL_SYMBOL = 1 << 6;
        /// #[no_debug], indicates that no debugging information should be
        /// generated for this function by LLVM
        const NO_DEBUG                  = 1 << 7;
        /// #[thread_local], indicates a static is actually a thread local
        /// piece of memory
        const THREAD_LOCAL              = 1 << 8;
        /// #[used], indicates that LLVM can't eliminate this function (but the
        /// linker can!)
        const USED                      = 1 << 9;
    }
}

impl CodegenFnAttrs {
    pub fn new() -> CodegenFnAttrs {
        CodegenFnAttrs {
            flags: CodegenFnAttrFlags::empty(),
            inline: InlineAttr::None,
            export_name: None,
            link_name: None,
            target_features: vec![],
            linkage: None,
            link_section: None,
        }
    }

    /// True if `#[inline]` or `#[inline(always)]` is present.
    pub fn requests_inline(&self) -> bool {
        match self.inline {
            InlineAttr::Hint | InlineAttr::Always => true,
            InlineAttr::None | InlineAttr::Never => false,
        }
    }

    /// True if it looks like this symbol needs to be exported, for example:
    ///
    /// * `#[no_mangle]` is present
    /// * `#[export_name(...)]` is present
    /// * `#[linkage]` is present
    pub fn contains_extern_indicator(&self) -> bool {
        self.flags.contains(CodegenFnAttrFlags::NO_MANGLE) ||
            self.export_name.is_some() ||
            match self.linkage {
                // these are private, make sure we don't try to consider
                // them external
                None |
                Some(Linkage::Internal) |
                Some(Linkage::Private) => false,
                Some(_) => true,
            }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Node<'a> {
    Item(&'a Item<'a>),
    ForeignItem(&'a ForeignItem<'a>),
    TraitItem(&'a TraitItem<'a>),
    ImplItem(&'a ImplItem<'a>),
    Variant(&'a Variant<'a>),
    Field(&'a StructField<'a>),
    AnonConst(&'a AnonConst),
    Expr(&'a Expr<'a>),
    Stmt(&'a Stmt<'a>),
    PathSegment(&'a PathSegment<'a>),
    Ty(&'a Ty<'a>),
    TraitRef(&'a TraitRef<'a>),
    Binding(&'a Pat<'a>),
    Pat(&'a Pat<'a>),
    Block(&'a Block<'a>),
    Local(&'a Local<'a>),
    MacroDef(&'a MacroDef<'a>),

    /// StructCtor represents a tuple struct.
    StructCtor(&'a VariantData<'a>),

    Lifetime(&'a Lifetime),
    GenericParam(&'a GenericParam<'a>),
    Visibility(&'a Visibility<'a>),

    Crate,
}
