cl_1 = 1;
cl_2 = 5;
Point(1) = {0, 0, 0, cl_2};
Point(2) = {30, 0, 0, cl_2};
Point(3) = {0, 240, 0, cl_2};
Point(4) = {30, 240, 0, cl_2};
Line(1) = {1, 2};
Line(2) = {2, 4};
Line(3) = {4, 3};
Line(4) = {3, 1};
Line Loop(12) = {3, 4, 1, 2};
Plane Surface(12) = {12};
Plane Surface(13) = {12};
Physical Surface(14) = {12};
Physical Surface(15) = {12};
Plane Surface(16) = {12};
Physical Surface(17) = {12};
Physical Surface(18) = {12};
Physical Surface(19) = {12};
Plane Surface(20) = {12};
Physical Surface(21) = {12};
