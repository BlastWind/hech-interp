{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ParallelListComp #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StarIsType #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE PatternSynonyms #-}

module GPT2.HListExtensions where

import Data.Kind (Type)
import GHC.TypeLits
import Unsafe.Coerce (unsafeCoerce)
import GHC.Exts (IsList (..))

data family HList (l :: [Type])

data instance HList '[] = HNil

data instance HList (x ': xs) = x `HCons` HList xs

pattern (:.) :: forall x (xs :: [Type]). x -> HList xs -> HList (x : xs)
pattern (:.) x xs = HCons x xs

type family HReplicateR (n :: Nat) (e :: Type) :: [Type] where
  HReplicateR 0 _ = '[]
  HReplicateR n e = e ': HReplicateR (n - 1) e

-- Ideally, we can write `hmap` as following:
-- However, because `HReplicateR` is non-injective, we don't necessarily know that
--  `HNil ~ HList (HReplicateR n a) => n = 0`.
-- Consequently, given `foo :: HList (HReplicateR n a)`, you simply can't
-- pattern match it. And hence, you can't write a `hmap`.
--
-- Ideally:
-- hmap :: (a -> b) -> HList (HReplicateR n a) -> HList (HReplicateR n b)
-- hmap _ _ = _
-- hmap f (HCons x xs) = HCons (f x) (hmap f xs)

-- So we will explicitly establish connections between type variables.
-- We will not let the compiler infer structure from `HReplicateR`,
-- turning
-- `HMapC` says: Given elements types `a` and `b`, and list structures `as` and `bs`,
-- we define a way to transform an `HList as` into an `HList bs` using a function `a -> b`.
class HMapC a b as bs | a b as -> bs, a b bs -> as where
  hmapC :: (a -> b) -> HList as -> HList bs

-- An empty list maps to an empty list, regardless of the element types
instance HMapC a b '[] '[] where
  hmapC _ HNil = HNil

-- If we know how to map a list of as to bs,
-- then we can map a list with an a at the front to a list with a b at the front
instance (HMapC a b as bs) => HMapC a b (a ': as) (b ': bs) where
  hmapC f (HCons x xs) = HCons (f x) (hmapC f xs)

mapFst ::
  forall a b n.
  (HMapC (a, b) a (HReplicateR n (a, b)) (HReplicateR n a)) =>
  HList (HReplicateR n (a, b)) ->
  HList (HReplicateR n a)
mapFst = hmapC @(a, b) fst

l1 :: HList (HReplicateR 2 (Int, String))
l1 = HCons (10, "asdf") (HCons (20, "zcv") HNil)

l2 :: HList (HReplicateR 2 Int)
l2 = mapFst @Int @_ @2 l1

class HLast (xs :: [Type]) (y :: Type) | xs -> y where
  hLast :: HList xs -> y

instance HLast '[x] x where
  hLast (x `HCons` HNil) = x

instance (HLast (b ': xs) y) => HLast (a ': b ': xs) y where
  hLast (_ `HCons` xs) = hLast xs

class HTail (xs :: [Type]) (ys :: [Type]) | xs -> ys where
  hTail :: HList xs -> HList ys

instance HTail (a ': as) as where
  hTail (_ `HCons` xs) = xs

-- HScanlC applies a binary operation cumulatively over a list
-- The type signature mirrors scanl: (b -> a -> b) -> b -> [a] -> [b]
-- But for heterogeneous lists, we need to track the evolving accumulator type
class HScanlC a b as bs | a b as -> bs where
  hScanlC :: (b -> a -> b) -> b -> HList as -> HList bs

-- Base case: scanning an empty list produces a singleton list containing just the initial value
instance HScanlC a b '[] '[b] where
  hScanlC _ b HNil = HCons b HNil

instance (HScanlC a b as bs) => HScanlC a b (a ': as) (b ': bs) where
  hScanlC :: (b -> a -> b) -> b -> HList (a : as) -> HList (b : bs)
  hScanlC f acc (HCons x xs) =
    let newAcc = f acc x
     in HCons acc (hScanlC f newAcc xs)

-- HScanlC
--                      Integer Integer (HReplicateR n Int) (HReplicateR (1 + n) Int)

l3 = hScanlC @Int (+) 0 l2

ok = hTail l3

foo ::
  forall n.
  ( HScanlC
      Integer
      Integer
      (HReplicateR n Int)
      (HReplicateR (1 + n) Int)
  ) =>
  HList (HReplicateR n Int) ->
  HList (HReplicateR (n + 1) Int)
foo = hScanlC (+) 0


instance IsList (Maybe (HList '[(a :: Type)])) where
  type Item (Maybe (HList '[(a :: Type)])) = a
  fromList [x] = liftA2 (:.) (Just x) (Just HNil)
  fromList _ = Nothing
  toList Nothing = []
  toList (Just (x :. HNil)) = [x]

instance
  ( IsList (Maybe (HList (a ': as))),
    a ~ Item (Maybe (HList (a ': as)))
  ) =>
  IsList (Maybe (HList ((a :: Type) ': a ': as)))
  where
  type Item (Maybe (HList (a ': a ': as))) = a
  fromList (x : xs) = liftA2 (:.) (Just x) (fromList xs)
  fromList _ = Nothing
  toList Nothing = []
  toList (Just (x :. xs)) = x : toList (Just xs)