{-# LANGUAGE AllowAmbiguousTypes #-}
{- Enriching Torch.HList -}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE ParallelListComp #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Strict #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=0 #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}

module GPT2.HListExtensions
  ( HList,
    HReplicateR,
    HAppendListR,
    HAppendFD,
    ApplyAB,
    applyAB,
    HScanl,
    hTail,
    Nat2HNat,
    HNat2Nat,
    hLast,
    HReplicate,
    hScanl,
    hScanlTail, 
    hReplicate,
    hReplicateF
  )
where

import Data.HList hiding (ApplyAB, applyAB)
import GHC.TypeLits
import Prelude hiding (cos, exp, sin)

-- | TODO: Originally, ApplyAB doesn't have this fundep.
class ApplyAB f a b | f a -> b where
  applyAB :: f -> a -> b

-- class HLast (xs :: [Type]) (y :: Type) | xs -> y where
--   hLast :: HList xs -> y

-- instance HLast '[x] x where
--   hLast :: HList '[x] -> x
--   hLast (x `HCons` HNil) = x

-- instance (HLast (b ': xs) y) => HLast (a ': b ': xs) y where
--   hLast (_`HCons` xs) = hLast xs

-- class HScanr f z ls rs where
--   -- | Correspond to `scanr :: (a -> b -> b) -> b -> [a] -> [b]`
--   hScanr :: f -> z -> HList ls -> HList rs

-- instance (lz ~ '[z]) => HScanr f z '[] lz where
--   hScanr _ z _ = HCons (z, HNil)

-- instance
--   ( ApplyAB f (x, r) s,
--     HScanr f z xs (r ': rs),
--     srrs ~ (s ': r ': rs)
--   ) =>
--   HScanr f z (x ': xs) srrs
--   where
--   hScanr f z (HCons (x, xs)) =
--     case hScanr f z xs :: HList (r ': rs) of
--       HCons (r, rs) -> HCons (applyAB f (x, r) :: s, HCons (r, rs))

class HScanl f z ls rs where
  -- | Correspond to `scanl :: (b -> a -> b) -> b -> [a] -> [b]`
  hScanl :: f -> z -> HList ls -> HList rs

scanr' :: (a -> b -> b) -> b -> [a] -> [b]
scanr' _ ini [] = [ini]
scanr' f acc (x : xs) = f x y : ys
  where
    ys@(y : _) = scanr' f acc xs

scanl' :: ((a, b) -> b) -> b -> [a] -> [b]
scanl' f acc xs = acc : [f (x, y) | x <- xs | y <- scanl' f acc xs]

instance (lz ~ '[z]) => HScanl f z '[] lz where
  hScanl _ z _ = z `HCons` HNil

instance
  ( ApplyAB f (z, x) s,
    HScanl f s xs rs
  ) =>
  HScanl f z (x ': xs) (z ': rs)
  where
  hScanl f z (x `HCons` xs) =
    z `HCons` hScanl f (applyAB f (z, x) :: s) xs

class HScanlTail f z ls rs where
  -- | Correspond to `scanl :: (b -> a -> b) -> b -> [a] -> [b]`
  hScanlTail :: f -> z -> HList ls -> HList rs

instance (lz ~ '[]) => HScanlTail f z '[] lz where
  hScanlTail _ _ _ = HNil

instance
  ( ApplyAB f (z, x) s,
    HScanlTail f s xs rs
  ) =>
  HScanlTail f z (x ': xs) (z ': rs)
  where
  hScanlTail f z (x `HCons` xs) =
    z `HCons` hScanlTail f (applyAB f (z, x) :: s) xs

data AddFunc = AddFunc

instance ApplyAB AddFunc (String, Int) String where
  applyAB _ (x, y) = show x ++ show y

type family Nat2HNat (n :: Nat) :: HNat where
  Nat2HNat 0 = HZero
  Nat2HNat n = HSucc (Nat2HNat (n - 1))

example :: IO ()
example = do
  -- Create an input list: [1, 2, 3]
  let inputList = (1 :: Int) .*. (2 :: Int) .*. HNil

      -- Apply hScanr with AddFunc, starting value 0
      result :: HList (HReplicateR (Nat2HNat 3) String)
      result = hScanl AddFunc "asdf" inputList

      ok = hTail result
  -- Result should be equivalent to scanr (+) 0 [1,2,3] = [6,5,3,0]
  print result
