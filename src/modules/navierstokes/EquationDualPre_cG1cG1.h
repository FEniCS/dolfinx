#ifndef __EQUATION_DUAL_PRE_CG1_CG1_HH
#define __EQUATION_DUAL_PRE_CG1_CG1_HH

class EquationDualPre_cG1cG1:public Equation {
      private:

	real theta;

	real delta1;
	real C1;

	real Re, dt, t;

	real u1sum, u2sum, u3sum, psum, divU, divUmid, normU;

	real umid1_1, umid1_2, umid1_3, umid1_4;
	real umid2_1, umid2_2, umid2_3, umid2_4;
	real umid3_1, umid3_2, umid3_3, umid3_4;
	real umid1, umid2, umid3;

	real D1umid1, D2umid1, D3umid1, D1umid2, D2umid2, D3umid2, D1umid3,
	    D2umid3, D3umid3;

	real Ii;
	real Iii;
	real Iij;

      public:

	//-----------------------------------------
	//      User definitions of equations
	//-----------------------------------------
	 EquationDualPre_cG1cG1():Equation(1, 3) {

		pi = 3.141592653589793238462;

		theta = 0.5;

		C1 = 1.0;

		Ii = 0.25;
		Iii = 0.1;
		Iij = 0.05;
	} ~EquationDualPre_cG1cG1() {
	}


	//--------- Equation definition -----------
	inline void jacobian(int el, int gpt, int tst_bf, int trial_bf,
			     int comp) {

		BasisFcn & p = *trialFcns[0];
		BasisFcn & q = *testFcns[0];

		real LAP = p.x() * q.x() + p.y() * q.y() + p.z() * q.z();

		work[0] = LAP;

	}


	//--------- Load definition --------------
	inline void residual(int el, int gpt, int tst_bf) {
	}




	//--------- Load definition for exact integration on tetrahedrons --------------
	inline
	    void jacobianExactInt(MV_Vector < real > &jac, int &tst_bf,
				  int &trial_bf, int &j_eq, int &el,
				  ShapeFunction *sfcn) {

    int cellsize = sfcn->GetNoNodes();

		if ((tst_bf == 0) && (trial_bf == 0)) {

			h = 2.0 * sfcn->GetCircumRadius();

			real qx, qy, qz;
			if (tst_bf == 0) {
				qx = sfcn->dN1x();
				qy = sfcn->dN1y();
				qz = sfcn->dN1z();
			} else if (tst_bf == 1) {
				qx = sfcn->dN2x();
				qy = sfcn->dN2y();
				qz = sfcn->dN2z();
			} else if (tst_bf == 2) {
				qx = sfcn->dN3x();
				qy = sfcn->dN3y();
				qz = sfcn->dN3z();
			} else if (tst_bf == 3) {
				qx = sfcn->dN4x();
				qy = sfcn->dN4y();
				qz = sfcn->dN4z();
			}

			real px, py, pz;
			if (trial_bf == 0) {
				px = sfcn->dN1x();
				py = sfcn->dN1y();
				pz = sfcn->dN1z();
			} else if (trial_bf == 1) {
				px = sfcn->dN2x();
				py = sfcn->dN2y();
				pz = sfcn->dN2z();
			} else if (trial_bf == 2) {
				px = sfcn->dN3x();
				py = sfcn->dN3y();
				pz = sfcn->dN3z();
			} else if (trial_bf == 3) {
				px = sfcn->dN4x();
				py = sfcn->dN4y();
				pz = sfcn->dN4z();
			}

			jac(0) = px * qx + py * qy + pz * qz;

		}

	}

	//--------- Load definition for exact integration on tetrahedrons --------------
	inline
	    void residualExactInt(MV_ColMat < real > &bres, int &bf,
				  int &el, ShapeFunction *sfcn) {

    int cellsize = sfcn->GetNoNodes();


    		if (bf == 0) {

			h = 2.0 * sfcn->GetCircumRadius();

			x(0) = sfcn->x1();
			x(1) = sfcn->x2();
			x(2) = sfcn->x3();
			x(3) = sfcn->x4();

			y(0) = sfcn->y1();
			y(1) = sfcn->y2();
			y(2) = sfcn->y3();
			y(3) = sfcn->y4();

			z(0) = sfcn->z1();
			z(1) = sfcn->z2();
			z(2) = sfcn->z3();
			z(3) = sfcn->z4();

			for (int i = 0; i < cellsize; i++)
				uprev_TS1(i) =
				    COEFF(0)->GetNodalValue(sfcn->
							    GetNode(i));
			sfcn->EvalGradient(uprev_TS1, uprev_TS1Grad);
			for (int i = 0; i < cellsize; i++)
				uprev_TS2(i) =
				    COEFF(1)->GetNodalValue(sfcn->
							    GetNode(i));
			sfcn->EvalGradient(uprev_TS2, uprev_TS2Grad);
			for (int i = 0; i < cellsize; i++)
				uprev_TS3(i) =
				    COEFF(2)->GetNodalValue(sfcn->
							    GetNode(i));
			sfcn->EvalGradient(uprev_TS3, uprev_TS3Grad);
			//for (int i=0;i<cellsize;i++) pprev_TS(i) = COEFF(3) -> GetNodalValue( sfcn->GetNode(i) );    
			//sfcn->EvalGradient(pprev_TS,pprev_TSGrad);

			for (int i = 0; i < cellsize; i++)
				U1(i) =
				    COEFF(4)->GetNodalValue(sfcn->
							    GetNode(i));
			sfcn->EvalGradient(U1, U1Grad);
			for (int i = 0; i < cellsize; i++)
				U2(i) =
				    COEFF(5)->GetNodalValue(sfcn->
							    GetNode(i));
			sfcn->EvalGradient(U2, U2Grad);
			for (int i = 0; i < cellsize; i++)
				U3(i) =
				    COEFF(6)->GetNodalValue(sfcn->
							    GetNode(i));
			sfcn->EvalGradient(U3, U3Grad);

			for (int i = 0; i < cellsize; i++)
				u1(i) =
				    COEFF(8)->GetNodalValue(sfcn->
							    GetNode(i));
			sfcn->EvalGradient(u1, u1Grad);
			for (int i = 0; i < cellsize; i++)
				u2(i) =
				    COEFF(9)->GetNodalValue(sfcn->
							    GetNode(i));
			sfcn->EvalGradient(u2, u2Grad);
			for (int i = 0; i < cellsize; i++)
				u3(i) =
				    COEFF(10)->GetNodalValue(sfcn->
							     GetNode(i));
			sfcn->EvalGradient(u3, u3Grad);
			for (int i = 0; i < cellsize; i++)
				p(i) =
				    COEFF(11)->GetNodalValue(sfcn->
							     GetNode(i));
			sfcn->EvalGradient(p, pGrad);

			dt = COEFF(12)->Eval();
			//t = COEFF(13) -> Eval();
			Re = COEFF(14)->Eval();


			divUmid = 0.5 * (uprev_TS1Grad(0) + u1Grad(0))
			    + 0.5 * (uprev_TS2Grad(1) + u2Grad(1))
			    + 0.5 * (uprev_TS3Grad(2) + u3Grad(2));

			u1sum = u1(0) + u1(1) + u1(2) + u1(3);
			u2sum = u2(0) + u2(1) + u2(2) + u2(3);
			u3sum = u3(0) + u3(1) + u3(2) + u3(3);
			normU =
			    sqrt(u1sum * u1sum + u2sum * u2sum +
				 u3sum * u3sum) / 4;

			if (h * Re > 1.0) {
				//delta1 = 0.5 / sqrt( 1.0/sqr(dt) + sqr(normU/h) );
				delta1 = C1 * h;
			} else {
				delta1 = C1 * sqr(h);
			}

			D1umid1 =
			    theta * uprev_TS1Grad(0) + (1.0 -
							theta) * u1Grad(0);
			D2umid1 =
			    theta * uprev_TS1Grad(1) + (1.0 -
							theta) * u1Grad(1);
			D3umid1 =
			    theta * uprev_TS1Grad(2) + (1.0 -
							theta) * u1Grad(2);

			D1umid2 =
			    theta * uprev_TS2Grad(0) + (1.0 -
							theta) * u2Grad(0);
			D2umid2 =
			    theta * uprev_TS2Grad(1) + (1.0 -
							theta) * u2Grad(1);
			D3umid2 =
			    theta * uprev_TS2Grad(2) + (1.0 -
							theta) * u2Grad(2);

			D1umid3 =
			    theta * uprev_TS3Grad(0) + (1.0 -
							theta) * u3Grad(0);
			D2umid3 =
			    theta * uprev_TS3Grad(1) + (1.0 -
							theta) * u3Grad(1);
			D3umid3 =
			    theta * uprev_TS3Grad(2) + (1.0 -
							theta) * u3Grad(2);

			umid1_1 =
			    theta * uprev_TS1(0) + (1.0 - theta) * u1(0);
			umid1_2 =
			    theta * uprev_TS1(1) + (1.0 - theta) * u1(1);
			umid1_3 =
			    theta * uprev_TS1(2) + (1.0 - theta) * u1(2);
			umid1_4 =
			    theta * uprev_TS1(3) + (1.0 - theta) * u1(3);

			umid2_1 =
			    theta * uprev_TS2(0) + (1.0 - theta) * u2(0);
			umid2_2 =
			    theta * uprev_TS2(1) + (1.0 - theta) * u2(1);
			umid2_3 =
			    theta * uprev_TS2(2) + (1.0 - theta) * u2(2);
			umid2_4 =
			    theta * uprev_TS2(3) + (1.0 - theta) * u2(3);

			umid3_1 =
			    theta * uprev_TS3(0) + (1.0 - theta) * u3(0);
			umid3_2 =
			    theta * uprev_TS3(1) + (1.0 - theta) * u3(1);
			umid3_3 =
			    theta * uprev_TS3(2) + (1.0 - theta) * u3(2);
			umid3_4 =
			    theta * uprev_TS3(3) + (1.0 - theta) * u3(3);

			umid1 = umid1_1 + umid1_2 + umid1_3 + umid1_4;
			umid2 = umid2_1 + umid2_2 + umid2_3 + umid2_4;
			umid3 = umid3_1 + umid3_2 + umid3_3 + umid3_4;
		}

		real qx, qy, qz;
		if (bf == 0) {
			qx = sfcn->dN1x();
			qy = sfcn->dN1y();
			qz = sfcn->dN1z();
		} else if (bf == 1) {
			qx = sfcn->dN2x();
			qy = sfcn->dN2y();
			qz = sfcn->dN2z();
		} else if (bf == 2) {
			qx = sfcn->dN3x();
			qy = sfcn->dN3y();
			qz = sfcn->dN3z();
		} else if (bf == 3) {
			qx = sfcn->dN4x();
			qy = sfcn->dN4y();
			qz = sfcn->dN4z();
		}

		real UDu_Dq_1 =
		    ((U1(0) + U1(1) + U1(2) + U1(3)) * D1umid1 +
		     (U2(0) + U2(1) + U2(2) + U2(3)) * D2umid1 + (U3(0) +
								  U3(1) +
								  U3(2) +
								  U3(3)) *
		     D3umid1) * Ii * qx;
		real UDu_Dq_2 =
		    ((U1(0) + U1(1) + U1(2) + U1(3)) * D1umid2 +
		     (U2(0) + U2(1) + U2(2) + U2(3)) * D2umid2 + (U3(0) +
								  U3(1) +
								  U3(2) +
								  U3(3)) *
		     D3umid2) * Ii * qy;
		real UDu_Dq_3 =
		    ((U1(0) + U1(1) + U1(2) + U1(3)) * D1umid3 +
		     (U2(0) + U2(1) + U2(2) + U2(3)) * D2umid3 + (U3(0) +
								  U3(1) +
								  U3(2) +
								  U3(3)) *
		     D3umid3) * Ii * qz;

		real DUu_Dq_1 =
		    (umid1 * U1Grad(0) + umid2 * U2Grad(0) +
		     umid3 * U3Grad(0)) * Ii * qx;
		real DUu_Dq_2 =
		    (umid1 * U1Grad(1) + umid2 * U2Grad(1) +
		     umid3 * U3Grad(1)) * Ii * qy;
		real DUu_Dq_3 =
		    (umid1 * U1Grad(2) + umid2 * U2Grad(2) +
		     umid3 * U3Grad(2)) * Ii * qz;


		real f1 = 0.0;
		real f2 = 0.0;
		real f3 = 0.0;

		real d = 0.125;
		real tf = 10.0 - d;
		//tf = 9.0;

		real centerpt_x = 1.5;
		real centerpt_y = 0.5;
		real centerpt_z = 0.5;
		real x_el = 0.25 * (x(0) + x(1) + x(2) + x(3));
		real y_el = 0.25 * (y(0) + y(1) + y(2) + y(3));
		real z_el = 0.25 * (z(0) + z(1) + z(2) + z(3));
		if ((t >= tf) && ((centerpt_x - 0.5 * d) <= x_el)
		    && ((centerpt_x + 0.5 * d) >= x_el)
		    && ((centerpt_y - 0.5 * d) <= y_el)
		    && ((centerpt_y + 0.5 * d) >= y_el)
		    && ((centerpt_z - 0.5 * d) <= z_el)
		    && ((centerpt_z + 0.5 * d) >= z_el)) {
			f1 = 1.0;
			f2 = 0.0;
			f3 = 0.0;
		}

		real f_Dq = (f1 * qx + f2 * qy + f3 * qz) * Ii;

		bres(0, bf) =
		    f_Dq + (UDu_Dq_1 + UDu_Dq_2 + UDu_Dq_3) - (DUu_Dq_1 +
							       DUu_Dq_2 +
							       DUu_Dq_3) -
		    (1.0 / delta1) * (divUmid * Ii);

	}


	/*
	   inline
	   void MatrixVectorProductExactInt( MV_ColMat<real>& bres, int& bf, int& el, FET4n3D& FE ){}

	   inline
	   void NonLinearResidualExactInt( MV_ColMat<real>& bres, int& bf, int& el, FET4n3D& FE ){}
	 */
};

#endif
