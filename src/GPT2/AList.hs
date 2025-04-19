{- A type-level association list (alist) where keys are symbols (type level strings).
Two operations supported — insert and lookup. 
When looking up, if your key isn't in the list, a compile-time error is raised.
-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE OverlappingInstances #-}

module GPT2.AList (module GPT2.AList) where
import Data.Kind
import GHC.TypeLits
import Prelude hiding (lookup)

-- | A type-level association list: Keys are `Symbol`s at the type-level!
data AssocList (alist :: [(Symbol, Type)]) where
  Empty :: AssocList '[]
  Cons :: v -> AssocList rest -> AssocList ('(k, v) ': rest)

-- | Closed type family specifying lookup behavior at the type level
type family LookupResult (key :: Symbol) (alist :: [(Symbol, Type)]) where
  LookupResult key '[] = TypeError ('Text "Key not found in associative list")
  LookupResult key ('(key, value) ': rest) = value
  LookupResult key ('(key', value) ': rest) = LookupResult key rest

-- | `Lookup` class allows us to associate a value-level `lookup` with the type level `LookupResult`
class Lookup (key :: Symbol) (alist :: [(Symbol, Type)]) where
  lookup :: AssocList alist -> LookupResult key alist

-- | Note, this instance doesn't get used: The TypeError in `LookupResult '[]` takes over.
instance Lookup key '[] where
  lookup _ = error "Key not found. (You won't see this error though, `LookupResult key '[]` computes a compile-time error.)"

instance Lookup key ('(key, value) ': rest) where
  lookup (Cons v _) = v

instance (Lookup key rest, LookupResult key ('(key', value) ': rest) ~ LookupResult key rest) 
      => Lookup key ('(key', value) ': rest) where
  lookup (Cons _ rest) = lookup @key rest

-- | Closed type family specifying insert behavior at the type level
-- Note, insertion is rightmost: You might need to check the full list,
-- by the time you have traversed the list, you might as well insert there,
-- towards the end
type family InsertResult (key :: Symbol) (value :: Type) (alist :: [(Symbol, Type)]) :: [(Symbol, Type)] where
  InsertResult key value '[] = '[ '(key, value) ]
  InsertResult key value ('(key, _) ': rest) = '(key, value) ': rest
  InsertResult key value ('(key', value') ': rest) = '(key', value') ': InsertResult key value rest

-- | `Insert` class allows us to associate a value-level `insert` with the type level `InsertResult`.
class Insert (key :: Symbol) (value :: Type) (alist :: [(Symbol, Type)]) where
  insert :: value -> AssocList alist -> AssocList (InsertResult key value alist)

instance Insert key value '[] where
  insert :: value -> AssocList '[] -> AssocList (InsertResult key value '[])
  insert v Empty = Cons v Empty

-- | When the key matches the first element's key, replace!
instance (InsertResult key value ('(key, value') ': rest) ~ ('(key, value) ': rest)) 
      => Insert key value ('(key, value') ': rest) where
  insert :: value -> AssocList ('(key, value') ': rest) -> AssocList (InsertResult key value ('(key, value') ': rest))
  insert v (Cons _ rest) = Cons v rest

-- | When the key doesn't match the first element's key, call insert on the tail
instance (InsertResult key value ('(key', value') ': rest) ~ ('(key', value') ': InsertResult key value rest), 
          Insert key value rest) 
      => Insert key value ('(key', value') ': rest) where
  insert :: value -> AssocList ('(key', value') ': rest) -> AssocList (InsertResult key value ('(key', value') ': rest))
  insert v (Cons v' rest) = Cons v' (insert @key v rest)

emptyAList :: AssocList '[]
emptyAList = Empty

-- a = insert @"asdf" "zcxv" $ insert @"zcv" 4 $ insert @"asdf" True emp

-- b = lookup @"asdf" a