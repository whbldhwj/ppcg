C(c0, c1, c2) = select(c2 == 0, 0, C(c0, c1, c2));
C(c0, c1, c2) = select(c2 == 0, (C(c0, c1, c2) + (A(c0, c1, c2) * B(c0, c1, c2))), C(c0, c1, c2));
C(c0, c1, c2) = select(c2 + -1 >= 0, (C(c0, c1, c2 - 1) + (A(c0, c1, c2) * B(c0, c1, c2))), C(c0, c1, c2));

