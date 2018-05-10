r = DefineNumber[0.25];
Point(1) = {-1, -1, -1, r};
Point(2) = {1, -1, -1, r};
Point(3) = {1, 1, -1, r};
Point(4) = {-1, 1, -1, r};
Point(5) = {-1, 1, 1, r};
Point(6) = {-1, -1, 1, r};
Point(10) = {1, -1, 1, r};
Point(14) = {1, 1, 1, r};
Line(1) = {4, 3};
Line(2) = {2, 3};
Line(3) = {4, 1};
Line(4) = {1, 2};
Line(8) = {5, 6};
Line(9) = {6, 10};
Line(10) = {10, 14};
Line(11) = {14, 5};
Line(13) = {4, 5};
Line(14) = {1, 6};
Line(18) = {2, 10};
Line(22) = {3, 14};
Line Loop(6) = {3, 4, 2, -1};
Plane Surface(6) = {6};
Line Loop(15) = {3, 14, -8, -13};
Ruled Surface(15) = {15};
Line Loop(19) = {4, 18, -9, -14};
Ruled Surface(19) = {19};
Line Loop(23) = {2, 22, -10, -18};
Ruled Surface(23) = {23};
Line Loop(27) = {-1, 13, -11, -22};
Ruled Surface(27) = {27};
Line Loop(28) = {8, 9, 10, 11};
Plane Surface(28) = {28};
Surface Loop(1) = {6, 15, 19, 23, 27, 28};
Volume(1) = {1};

Physical Surface("Inflow",2) = {6,15, 19, 27, 28,23};
Physical Volume("Domain",9) = {1};
Coherence;
