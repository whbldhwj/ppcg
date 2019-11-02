A(c0, c1, c2) = 0;
A(c0, c1, c2) = select(c1 == 0, A[c0][c2], select(c1 + -1 >= 0, A(c0, c1 - 1, c2), A(c0, c1, c2)));
B(c0, c1, c2) = 0;
B(c0, c1, c2) = select(c0 == 0, B[c2][c1], select(c0 + -1 >= 0, B(c0 - 1, c1, c2), B(c0, c1, c2)));
C(c0, c1, c2) = select(c2 == 0, 0, C(c0, c1, c2));
C(c0, c1, c2) = select(c2 == 0, (C(c0, c1, c2) + (A(c0, c1, c2) * B(c0, c1, c2))), C(c0, c1, c2));
C(c0, c1, c2) = select(c2 + -1 >= 0, (C(c0, c1, c2 - 1) + (A(c0, c1, c2) * B(c0, c1, c2))), C(c0, c1, c2));

