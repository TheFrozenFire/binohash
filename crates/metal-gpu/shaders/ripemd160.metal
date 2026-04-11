// ============================================================
// RIPEMD-160
// ============================================================

inline uint rotl(uint x, uint n) { return (x << n) | (x >> (32 - n)); }

void ripemd160_32bytes(thread const uchar* input, thread uchar* output) {
    uint W[16];
    for (int i = 0; i < 8; i++)
        W[i] = ((uint)input[i*4]) | ((uint)input[i*4+1]<<8) |
               ((uint)input[i*4+2]<<16) | ((uint)input[i*4+3]<<24);
    W[8] = 0x00000080;
    for (int i = 9; i < 14; i++) W[i] = 0;
    W[14] = 256; W[15] = 0;

    uint h0=0x67452301,h1=0xEFCDAB89,h2=0x98BADCFE,h3=0x10325476,h4=0xC3D2E1F0;

    const uint rl[80] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
        3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
        1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
        4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
    const uint sl[80] = {
        11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
        7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
        11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
        11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
        9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
    const uint rr[80] = {
        5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
        6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
        15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
        8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
        12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
    const uint sr[80] = {
        8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
        9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
        9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
        15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
        8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
    const uint kl[5] = {0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
    const uint kr[5] = {0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};

    uint al=h0,bl=h1,cl=h2,dl=h3,el=h4;
    uint ar=h0,br=h1,cr=h2,dr=h3,er=h4;

    for (int j = 0; j < 80; j++) {
        uint fl, fr;
        int rnd = j / 16;
        switch(rnd) {
            case 0: fl=bl^cl^dl;         fr=br^(cr|~dr);   break;
            case 1: fl=(bl&cl)|(~bl&dl);  fr=(br&dr)|(cr&~dr); break;
            case 2: fl=(bl|~cl)^dl;       fr=(br|~cr)^dr;   break;
            case 3: fl=(bl&dl)|(cl&~dl);  fr=(br&cr)|(~br&dr); break;
            default: fl=bl^(cl|~dl);      fr=br^cr^dr;      break;
        }
        uint tl = rotl(al+fl+W[rl[j]]+kl[rnd], sl[j])+el;
        al=el;el=dl;dl=rotl(cl,10);cl=bl;bl=tl;
        uint tr = rotl(ar+fr+W[rr[j]]+kr[rnd], sr[j])+er;
        ar=er;er=dr;dr=rotl(cr,10);cr=br;br=tr;
    }

    uint t = h1+cl+dr;
    h1=h2+dl+er; h2=h3+el+ar; h3=h4+al+br; h4=h0+bl+cr; h0=t;

    uint hh[5] = {h0,h1,h2,h3,h4};
    for (int i = 0; i < 5; i++) {
        output[i*4]=(uchar)(hh[i]); output[i*4+1]=(uchar)(hh[i]>>8);
        output[i*4+2]=(uchar)(hh[i]>>16); output[i*4+3]=(uchar)(hh[i]>>24);
    }
}

void hash160_33bytes(thread const uchar* input, thread uchar* output) {
    uint W[64];
    for (int i = 0; i < 8; i++)
        W[i] = ((uint)input[i*4]<<24)|((uint)input[i*4+1]<<16)|
               ((uint)input[i*4+2]<<8)|(uint)input[i*4+3];
    W[8] = ((uint)input[32]<<24) | 0x800000;
    for (int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 264;
    uint state[8];
    for (int i = 0; i < 8; i++) state[i] = SHA256_H0[i];
    sha256_compress(state, W);
    uchar sha_out[32];
    for (int i = 0; i < 8; i++) {
        sha_out[i*4]=(uchar)(state[i]>>24); sha_out[i*4+1]=(uchar)(state[i]>>16);
        sha_out[i*4+2]=(uchar)(state[i]>>8); sha_out[i*4+3]=(uchar)(state[i]);
    }
    ripemd160_32bytes(sha_out, output);
}

