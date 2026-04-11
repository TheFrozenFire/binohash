// ============================================================
// EC point operations (Jacobian coordinates)
// ============================================================

struct JacobianPoint { uint256 x, y, z; };
struct AffinePoint  { uint256 x, y; };

JacobianPoint ec_add_mixed(JacobianPoint p, AffinePoint q) {
    if (uint256_is_zero(p.z))
        return JacobianPoint{q.x, q.y, uint256_one()};

    uint256 z2 = field_sqr(p.z);
    uint256 u2 = field_mul(q.x, z2);
    uint256 z3 = field_mul(z2, p.z);
    uint256 s2 = field_mul(q.y, z3);
    uint256 h = field_sub(u2, p.x);
    uint256 r = field_sub(s2, p.y);

    if (uint256_is_zero(h)) {
        if (uint256_is_zero(r)) {
            uint256 a = field_sqr(p.y);
            uint256 b4 = field_mul(p.x, a);
            b4 = field_add(b4, b4); b4 = field_add(b4, b4);
            uint256 c = field_sqr(a); c = field_add(c, c);
            uint256 dd = field_sqr(p.x);
            uint256 e = field_add(dd, field_add(dd, dd));
            uint256 nx = field_sub(field_sqr(e), field_add(b4, b4));
            uint256 ny = field_sub(field_mul(e, field_sub(b4, nx)), field_add(c, c));
            uint256 nz = field_mul(p.y, p.z); nz = field_add(nz, nz);
            return JacobianPoint{nx, ny, nz};
        }
        return JacobianPoint{uint256_zero(), uint256_zero(), uint256_zero()};
    }

    uint256 h2 = field_sqr(h);
    uint256 h3 = field_mul(h2, h);
    uint256 u1h2 = field_mul(p.x, h2);
    uint256 nx = field_sub(field_sub(field_sqr(r), h3), field_add(u1h2, u1h2));
    uint256 ny = field_sub(field_mul(r, field_sub(u1h2, nx)), field_mul(p.y, h3));
    uint256 nz = field_mul(h, p.z);
    return JacobianPoint{nx, ny, nz};
}

AffinePoint jacobian_to_affine(JacobianPoint p) {
    uint256 zi = field_inv(p.z);
    uint256 z2 = field_sqr(zi);
    uint256 z3 = field_mul(z2, zi);
    return AffinePoint{field_mul(p.x, z2), field_mul(p.y, z3)};
}

JacobianPoint ec_mul_gtable(uint256 scalar,
    const device uchar* gtx, const device uchar* gty)
{
    ushort chunks[16];
    for (int i = 0; i < 8; i++) {
        chunks[i*2]     = (ushort)(scalar.d[i] & 0xFFFF);
        chunks[i*2 + 1] = (ushort)(scalar.d[i] >> 16);
    }
    JacobianPoint result = {uint256_zero(), uint256_zero(), uint256_zero()};
    for (int chunk = 0; chunk < 16; chunk++) {
        if (chunks[chunk] == 0) continue;
        uint index = ((uint)chunk * 65536 + (uint)(chunks[chunk] - 1)) * 32;
        AffinePoint gp = {uint256_from_be_device(gtx + index),
                          uint256_from_be_device(gty + index)};
        result = ec_add_mixed(result, gp);
    }
    return result;
}

