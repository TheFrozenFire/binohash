use ripemd::Ripemd160;
use sha2::{Digest, Sha256};

pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(&Sha256::digest(data));
    out
}

pub fn sha256d(data: &[u8]) -> [u8; 32] {
    sha256(&sha256(data))
}

pub fn ripemd160(data: &[u8]) -> [u8; 20] {
    let mut out = [0u8; 20];
    out.copy_from_slice(&Ripemd160::digest(data));
    out
}

pub fn hash160(data: &[u8]) -> [u8; 20] {
    ripemd160(&sha256(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_empty() {
        let hash = sha256(b"");
        assert_eq!(
            hex::encode(hash),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256d_empty() {
        let hash = sha256d(b"");
        // SHA256d("") = SHA256(SHA256(""))
        let expected = sha256(&sha256(b""));
        assert_eq!(hash, expected);
    }

    #[test]
    fn ripemd160_empty() {
        let hash = ripemd160(b"");
        assert_eq!(
            hex::encode(hash),
            "9c1185a5c5e9fc54612808977ee8f548b2258d31"
        );
    }

    #[test]
    fn hash160_is_ripemd160_of_sha256() {
        let data = b"hello world";
        assert_eq!(hash160(data), ripemd160(&sha256(data)));
    }
}
