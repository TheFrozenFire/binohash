// ============================================================
// DER check
// ============================================================

bool check_der_20(thread const uchar* d) {
    if (d[0]!=0x30) return false;
    int tl=d[1]; if (tl+3!=20) return false;
    int idx=2;
    for (int p=0;p<2;p++) {
        if (idx>=19||d[idx]!=0x02) return false; idx++;
        int il=d[idx]; idx++;
        if (il==0||idx+il>19) return false;
        if (il>1&&d[idx]==0x00&&!(d[idx+1]&0x80)) return false;
        if (d[idx]&0x80) return false; idx+=il;
    }
    return idx==19;
}

bool check_der_easy(thread const uchar* d) { return (d[0]>>4)==0x3; }

// ============================================================
// Pinning search kernel
