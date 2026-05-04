{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module SafeTensors where

import Control.Exception.Safe (throwIO)
import Control.Monad (when)
import Data.Aeson
import qualified Data.Aeson.Key as K
import qualified Data.Aeson.KeyMap as K
import Data.Binary.Get
import qualified Data.ByteString.Internal as BSI
import qualified Data.ByteString.Lazy as B
import qualified Data.HashMap.Strict as HM
import qualified Data.Map as M
import qualified Foreign.ForeignPtr as F
import qualified Foreign.Marshal.Utils as FU
import qualified Foreign.Ptr as F
import GHC.Generics
import Torch hiding (shape)

-- | Public type: tensors keyed by name. The byte-offset ordering used during
--   parsing (see 'SafeTensorKey') is an internal concern — by the time
--   tensors land here, only name-based lookup matters, so we use a HashMap
--   for O(1) lookup.
type SafeTensors = HM.HashMap String Tensor

class SafeTensorsModel a where
  load :: SafeTensors -> a

newtype Metadata = Metadata
  { format :: String
  }
  deriving (Show, Generic)

instance FromJSON Metadata

-- ----

data TensorData = TensorData
  { dtype :: !String,
    shape :: ![Int],
    data_offsets :: ![Int]
  }
  deriving (Show, Generic, Eq)

instance Ord TensorData where
  x <= y = head (data_offsets x) <= head (data_offsets y)

instance FromJSON TensorData

-- ----
-- | Internal key type used during the parse pass only. The point of carrying
--   the byte offset is so that the @Ord@ instance sorts entries by file
--   offset, which makes 'readTensors' walk the underlying ByteString
--   sequentially (each tensor's bytes follow the previous one's). Outside the
--   parser, tensors are re-keyed by name — see 'SafeTensors'.
data SafeTensorKey = SafeTensorKey {key :: !String, value :: !Int}
  deriving (Show)

instance Eq SafeTensorKey where
  x == y = key x == key y

instance Ord SafeTensorKey where
  x <= y = value x <= value y

lookupWithSafeTensorKey :: String -> SafeTensors -> Maybe Tensor
lookupWithSafeTensorKey = HM.lookup

-- ----

data SafeTensorMetadata = SafeTensorMetadata
  { metadata :: !Metadata,
    tensorData :: !(M.Map SafeTensorKey TensorData)
  }
  deriving (Show, Generic)

instance FromJSON SafeTensorMetadata where
  parseJSON = withObject "SafeTensorMetadata" $ \o -> do
    m <- o .: "__metadata__"
    t <- mapM parseJSON $ K.filterWithKey (\k _ -> k /= "__metadata__") o
    return $ SafeTensorMetadata m (M.fromList $ map (\(k, v) -> (SafeTensorKey (K.toString k) (head $ data_offsets v), v)) $ K.toList t)

-- ----

createTensor :: [Int] -> Int -> B.ByteString -> IO Tensor
createTensor shape' len v = do
  withTensor t $ \ptr1 -> do
    let (BSI.PS fptr _ len') = B.toStrict v
    when (len' < len) $ do
      throwIO $ userError $ "Read data's size is less than input tensor's one(" <> show len <> ")."
    F.withForeignPtr fptr $ \ptr2 -> do
      FU.copyBytes (F.castPtr ptr1) (F.castPtr ptr2) (Prelude.min len len')
  return t
  where
    t = zeros' shape'

getTensor :: TensorData -> Get (IO Tensor)
getTensor td = do
  v <- getLazyByteString $ fromIntegral len
  return $ createTensor (shape td) len v
  where
    len = last offsets - head offsets
    offsets = data_offsets td

readTensors :: SafeTensorMetadata -> Get (IO SafeTensors)
readTensors meta = do
  metaLen <- getInt64le
  _ <- skip $ fromIntegral metaLen
  -- 'mapM getTensor tmap' walks the map in ascending key order, which is
  -- ascending byte-offset order (see SafeTensorKey's Ord instance), so the
  -- underlying Get pulls tensor bytes sequentially. Once tensors are read,
  -- we re-key by name into a HashMap for the public API.
  tensorIOs <- mapM getTensor tmap
  return $ rekey <$> sequence tensorIOs
  where
    tmap = tensorData meta
    rekey = HM.fromList . map (\(k, v) -> (key k, v)) . M.toList

getMetadata :: Get B.ByteString
getMetadata = do
  metaLen <- getInt64le
  getLazyByteString metaLen

readSafeTensors :: FilePath -> IO SafeTensors
readSafeTensors path = do
  contents <- B.readFile path
  case decode (runGet getMetadata contents) :: Maybe SafeTensorMetadata of
    Nothing -> throwIO $ userError $ "Error with decoding contents of(" <> show path <> ")."
    Just meta -> runGet (readTensors meta) contents
