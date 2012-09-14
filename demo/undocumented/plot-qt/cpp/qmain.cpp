// Copyright (C) 2012 Joachim Berdal Haga
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2012-09-14
// Last changed: 2012-09-14
//
// This demo illustrates embedding the plot window in a Qt application.

#include <QtGui>

#include <dolfin.h>

#include "Plotter.h"

using namespace dolfin;

class VectorExpression : public Expression
{
public:

  VectorExpression() : Expression(2), t(0) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = -(x[1] - t)*exp(-10.0*(pow(x[0] - t, 2) + pow(x[1] - t, 2)));
    values[1] =  (x[0] - t)*exp(-10.0*(pow(x[0] - t, 2) + pow(x[1] - t, 2)));
  }

  double t;

};

int main(int argc, char *argv[])
{
  // Create application and top-level window
  QApplication app(argc, argv);
  QWidget window;
  window.setWindowTitle("Qt embedded plot window demo");

  // Create plotter for vector valued expression
  boost::shared_ptr<Mesh> unit_square(new UnitSquare(16, 16));
  boost::shared_ptr<VectorExpression> f_vector(new VectorExpression());
  Plotter plotter(f_vector, unit_square);

  // Create bottom row of labels/buttons
  QLabel *status_label = new QLabel;
  QLabel *x_label = new QLabel("x");
  QLabel *y_label = new QLabel("y");
  QPushButton *toggle = new QPushButton("Toggle mesh");

  QBoxLayout *sublayout = new QHBoxLayout();
  sublayout->addWidget(status_label);
  sublayout->addWidget(x_label);
  sublayout->addWidget(y_label);
  sublayout->addWidget(toggle);

  // Create main layout (the plot window above the row of labels/buttons)
  QBoxLayout *layout = new QVBoxLayout();
  layout->addWidget(plotter.get_widget());
  layout->addLayout(sublayout);
  window.setLayout(layout);

  // Connect the plotter with the labels and buttons
  QObject::connect(plotter.get_widget(), SIGNAL(mouseX(int)), x_label, SLOT(setNum(int)));
  QObject::connect(plotter.get_widget(), SIGNAL(mouseY(int)), y_label, SLOT(setNum(int)));
  QObject::connect(toggle, SIGNAL(pressed()), &plotter, SLOT(toggleMesh()));

  // Set window size and show window
  // FIXME: The plot window isn't correctly sized unless plot() has been called
  // before resize().
  plotter.plot();
  window.resize(700,500);
  window.show();

  // Plot a moving vector field. Since plot() calls qApp->processEvents(), user
  // interaction is possible while the plot moves.
  status_label->setText("Plotting...");
  for (dolfin::uint i = 0; i < 500; i++)
  {
    f_vector->t += 0.002;
    plotter.plot();
  }
  status_label->setText("Finished.");

  // Enter main event loop
  return app.exec();
}


