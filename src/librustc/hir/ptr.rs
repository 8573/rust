use std::fmt::{self, Display, Debug};
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};
use std::{mem, ptr, slice, vec};
use std::marker::PhantomData;
use arena::SyncDroplessArena;
use smallvec::SmallVec;

use serialize::{Encodable, Decodable, Encoder, Decoder};

use rustc_data_structures::stable_hasher::{StableHasher, StableHasherResult,
                                           HashStable};

pub trait IteratorExt: Iterator {
    fn collect_hir_vec(self, arena: &SyncDroplessArena) -> P<'_, [Self::Item]>;
}

impl<T: Iterator> IteratorExt for T where T::Item: Copy {
    fn collect_hir_vec(self, arena: &SyncDroplessArena) -> P<'_, [Self::Item]> {
        P::from_iter(arena, self)
    }
}

#[derive(Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct P<'a, T: ?Sized>(&'a T);

impl<'a, T: 'a+?Sized> Clone for P<'a, T> {
    #[inline]
    fn clone(&self) -> Self {
        P(self.0)
    }
}
impl<'a, T: 'a+?Sized> Copy for P<'a, T> {}

impl<'a, T: ?Sized> Deref for P<'a, T> {
    type Target = &'a T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T: Copy> P<'a, [T]> {
    #[inline]
    pub fn from_slice(arena: &'a SyncDroplessArena, slice: &[T]) -> Self where T: Clone {
        if slice.is_empty() {
            P::new()
        } else {
            P(arena.alloc_slice(slice))
        }
    }

    #[inline]
    pub fn from_iter<I: IntoIterator<Item=T>>(arena: &'a SyncDroplessArena, iter: I) -> Self
    where T: Clone {
        let mut iter = iter.into_iter();
        assert!(!mem::needs_drop::<T>());

        let size_hint = iter.size_hint();

        match size_hint {
            (min, Some(max)) if min == max => {
                if min == 0 {
                    return P::new();
                }
                let size = min.checked_mul(mem::size_of::<T>()).unwrap();
                let mem = arena.alloc_raw(size, mem::align_of::<T>()) as *mut _ as *mut T;
                unsafe {
                    for i in 0..min {
                        ptr::write(mem.offset(i as isize), iter.next().unwrap())
                    }
                    P(slice::from_raw_parts_mut(mem, min))
                }
            }
            (min, _) => {
                let vec: SmallVec<[_; 8]> = iter.collect();
                P::from_slice(arena, &vec)
            }
        }
    }
}

impl<'a, T: Copy> P<'a, T> {
    /// Equivalent to and_then(|x| x)
    #[inline]
    pub fn into_inner(&self) -> T {
        *self.0
    }

    /// Move out of the pointer.
    /// Intended for chaining transformations not covered by `map`.
    #[inline]
    pub fn and_then<U, F>(&self, f: F) -> U where
        F: FnOnce(T) -> U,
    {
        f(*self.0)
    }

    #[inline]
    pub fn alloc(arena: &'a SyncDroplessArena, inner: T) -> Self {
        P(arena.alloc(inner))
    }
}

impl<'a, T> P<'a, T> {
    #[inline]
    pub fn empty_thin() -> P<'a, P<'a, [T]>> {
        P(&P(&[]))
    }
}

impl<'a, T: ?Sized> P<'a, T> {
    // FIXME: Doesn't work with deserialization
    #[inline]
    pub fn from_existing(val: &'a T) -> Self {
        P(val)
    }
}

impl<T: ?Sized + Debug> Debug for P<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(self.0, f)
    }
}

impl<T: Display> Display for P<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&**self, f)
    }
}

impl<T> fmt::Pointer for P<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&(self.0 as *const T), f)
    }
}

impl<T: Decodable> Decodable for P<'_, T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
        panic!()
        //Decodable::decode(d).map(P)
    }
}

impl<T: Encodable> Encodable for P<'_, T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

impl<'a, T> P<'a, [T]> {
    #[inline]
    pub fn new() -> Self {
        P(&[])
    }

    #[inline(never)]
    pub fn into_vec(self) -> Vec<T> where T: Clone {
        (*self.0).iter().cloned().collect()
    }
}

impl<T> Default for P<'_, [T]> {
    /// Creates an empty `P<[T]>`.
    #[inline]
    fn default() -> Self {
        P::new()
    }
}

impl<T: Clone> Into<Vec<T>> for P<'_, [T]> {
    fn into(self) -> Vec<T> {
        self.into_vec()
    }
}

impl<T: Clone> IntoIterator for P<'_, [T]> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a P<'_, [T]> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T: Encodable> Encodable for P<'_, [T]> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        Encodable::encode(&**self, s)
    }
}

impl<T: Decodable> Decodable for P<'_, [T]> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
        panic!()//Ok(P::from_vec(Decodable::decode(d)?))
    }
}

impl<CTX, T> HashStable<CTX> for P<'_, T>
    where T: ?Sized + HashStable<CTX>
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        (**self).hash_stable(hcx, hasher);
    }
}
