{- Enriching Torch.HList -}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE Strict #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=0 #-}


module GPT2.HListExtensions (module GPT2.HListExtensions) where

import Data.Kind
import Torch.HList
import Prelude hiding (cos, exp, sin)


-- | No constraints on result and argument types
class ApplyAB f a b where
  applyAB :: f -> a -> b

class HLast (xs :: [Type]) (y :: Type) | xs -> y where
  hLast :: HList xs -> y

instance HLast '[x] x where
  hLast (x :. HNil) = x

instance (HLast (b ': xs) y) => HLast (a ': b ': xs) y where
  hLast (_ :. xs) = hLast xs

class HScanr f z ls rs where
  hScanr :: f -> z -> HList ls -> HList rs

instance (lz ~ '[z]) => HScanr f z '[] lz where
  hScanr _ z _ = HCons (z, HNil)

instance
  ( ApplyAB f (x, r) s,
    HScanr f z xs (r ': rs),
    srrs ~ (s ': r ': rs)
  ) =>
  HScanr f z (x ': xs) srrs
  where
  hScanr f z (HCons (x, xs)) =
    case hScanr f z xs :: HList (r ': rs) of
      HCons (r, rs) -> HCons (applyAB f (x, r) :: s, HCons (r, rs))
