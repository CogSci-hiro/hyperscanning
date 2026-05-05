{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::

   {% for item in methods %}
   {% if item != "__init__" and not item.startswith("_") %}
      ~{{ objname }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::

   {% for item in attributes %}
   {% if not item.startswith("_") %}
      ~{{ objname }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% if methods %}
   .. rubric:: Method Details

   {% for item in methods %}
   {% if item != "__init__" and not item.startswith("_") %}
   .. automethod:: {{ fullname }}.{{ item }}

   {% endif %}
   {%- endfor %}
   {% endif %}

   {% if attributes %}
   .. rubric:: Attribute Details

   {% for item in attributes %}
   {% if not item.startswith("_") %}
   .. autoattribute:: {{ fullname }}.{{ item }}

   {% endif %}
   {%- endfor %}
   {% endif %}
